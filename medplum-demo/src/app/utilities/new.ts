import {
  AIMessage,
  BaseMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import type { RunnableConfig } from "@langchain/core/runnables";
import { Runnable } from "@langchain/core/runnables";
import { StructuredTool, tool } from "@langchain/core/tools";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  Annotation,
  MemorySaver,
  NodeInterrupt,
  START,
  StateGraph,
} from "@langchain/langgraph/web";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";

// #region schemas

enum ResourceType {
  Patient = "Patient",
}

const humanNameSchema = z.object({
  use: z
    .enum([
      "usual",
      "official",
      "temp",
      "nickname",
      "anonymous",
      "old",
      "maiden",
    ])
    .describe("Identifies the purpose for this name."),
  family: z
    .string()
    .describe("The part of a name that links to the genealogy."),
  given: z.array(z.string()).describe("Given name."),
});

const narrativeSchema = z.object({
  status: z
    .enum(["generated", "extensions", "additional", "empty"])
    .describe(
      "The status of the narrative - whether it's entirely generated (from just the defined data or the extensions too), or whether a human authored it and it may contain additional data."
    ),
  div: z
    .string()
    .describe(
      "The actual narrative content, a stripped down version of XHTML."
    ),
});

const patientCreateSchema = z.object({
  resourceType: z.literal(ResourceType.Patient),
  text: narrativeSchema.describe(
    "A human-readable narrative that contains a summary of the resource and can be used to represent the content of the resource to a human."
  ),
  name: z
    .array(humanNameSchema)
    .describe("A name associated with the individual."),
  gender: z
    .enum(["male", "female", "other", "unknown"])
    .describe(
      "Administrative Gender - the gender that the patient is considered to have for administration and record keeping purposes."
    ),
  birthDate: z.string().describe("The date of birth for the individual."),
});

const patientSchema = patientCreateSchema.extend({
  id: z.string().describe("Logical id of this artifact."),
});

type PatientCreate = z.infer<typeof patientCreateSchema>;
type Patient = z.infer<typeof patientSchema>;

// #endregion schemas

async function createAgent({
  llm,
  tools,
  systemMessage,
}: {
  llm: ChatOpenAI;
  tools: StructuredTool[];
  systemMessage: string;
}) {
  const toolNames = tools.map((t) => t.name).join(", ");
  const formattedTools = tools.map(convertToOpenAITool);

  let prompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful AI EHR assistant, collaborating with other EHR assistants." +
        " Use the provided tools to progress towards answering the question." +
        " If you are unable to fully answer, that's OK, another assistant with different tools " +
        " will help where you left off. Execute what you can to make progress." +
        " You have access to the following tools: {tool_names}.\n{system_message}",
    ],
    new MessagesPlaceholder("messages"),
  ]);
  prompt = await prompt.partial({
    system_message: systemMessage,
    tool_names: toolNames,
  });

  return prompt.pipe(llm.bindTools(formattedTools, { strict: true }));
}

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  sender: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => "user",
  }),
  patientCreate: Annotation<PatientCreate | null>({
    reducer: (_, y) => y,
    default: () => null,
  }),
  patient: Annotation<Patient | null>({
    reducer: (_, y) => y,
    default: () => null,
  }),
  patientUpdate: Annotation<Patient | null>({
    reducer: (_, y) => y,
    default: () => null,
  }),
  resourceApproved: Annotation<boolean | null>({
    reducer: (_, y) => y,
    default: () => null,
  }),
});

// #region tools

const patientCreateTool = tool(
  (patientCreate) => {
    console.log(patientCreate);
    return {
      resourceApproved: null,
      patient: null,
      patientCreate,
      patientUpdate: null,
    };
  },
  {
    name: "patient_create",
    description: "Creates a new patient in the EHR system.",
    schema: patientCreateSchema,
  }
);

const patientUpdateTool = tool(
  (patientUpdate) => {
    console.log(patientUpdate);
    return {
      resourceApproved: null,
      patientCreate: null,
      patientUpdate,
    };
  },
  {
    name: "patient_update",
    description: "Updates an existing patient in the EHR system.",
    schema: patientSchema,
  }
);

const resourceApprovalTool = tool(
  ({ approved }) => {
    console.log("APPROVAL", approved);
    return {
      resourceApproved: approved,
    };
  },
  {
    name: "resource_draft",
    description: "Verifies the user input for a new resource.",
    schema: z.object({
      approved: z.boolean().describe("Whether the resource is approved."),
    }),
  }
);

// #endregion tools

// #region graph

async function runAgentNode({
  state,
  agent,
  name,
  config,
}: {
  state: typeof AgentState.State;
  agent: Runnable;
  name: string;
  config?: RunnableConfig;
}) {
  if (state.sender === name) {
    throw new NodeInterrupt("Wait for user input");
  }
  let result = await agent.invoke(state, config);
  //   if (!result?.tool_calls || result.tool_calls.length === 0) {
  //     result = new HumanMessage({ ...result, name: name });
  //   }
  return {
    ...state,
    messages: [result],
    sender: name,
  };
}

export async function createGraph(openAIApiKey: string) {
  const llm = new ChatOpenAI({ modelName: "gpt-4o", apiKey: openAIApiKey });
  const tools = [patientCreateTool, patientUpdateTool, resourceApprovalTool];
  const toolNode = new ToolNode<typeof AgentState.State>(tools);

  // #region agents

  const patientCreateAgent = await createAgent({
    llm,
    tools: [patientCreateTool],
    systemMessage:
      "You are an EHR assistant that helps create new patients in the system.",
  });

  const patientUpdateAgent = await createAgent({
    llm,
    tools: [patientUpdateTool],
    systemMessage:
      "You are an EHR assistant that helps update existing patients in the system.",
  });

  const resourceApprovalAgent = await createAgent({
    llm,
    tools: [resourceApprovalTool],
    systemMessage:
      "You are an assistant that verifies with the user the input for a new resource.",
  });

  // #endregion agents

  // #region nodes

  async function patientCreateNode(
    state: typeof AgentState.State,
    config?: RunnableConfig
  ) {
    return runAgentNode({
      state,
      agent: patientCreateAgent,
      name: "Patient creater",
      config,
    });
  }

  async function patientUpdateNode(
    state: typeof AgentState.State,
    config?: RunnableConfig
  ) {
    return runAgentNode({
      state,
      agent: patientUpdateAgent,
      name: "Patient updater",
      config,
    });
  }

  async function draftResourceApproveNode(
    state: typeof AgentState.State,
    config?: RunnableConfig
  ) {
    const prompt = new SystemMessage(
      `The AI assistant has initialized the new JSON resource: ${JSON.stringify(
        state.patientCreate ?? state.patientUpdate,
        null,
        2
      )}. Please ask the user if the resource is correct. Do not show any JSON and present the data to the user in a human-readable format.`
    );
    const result = await runAgentNode({
      state: { ...state, messages: [...state.messages, prompt] },
      agent: resourceApprovalAgent,
      name: "Draft resource approver",
      config,
    });
    return {
      ...result,
      messages: result.messages.slice(-1),
    };
  }

  async function approveResourceNode(
    state: typeof AgentState.State,
    _config?: RunnableConfig
  ) {
    return {
      patient: state.patientUpdate,
      patientCreate: null,
      patientUpdate: null,
      resourceApproved: null,
      sender: "Resource approver",
    };
  }

  async function toolMessageUpdateStateNode(
    state: typeof AgentState.State,
    config?: RunnableConfig
  ) {
    const lastMessage = state.messages[
      state.messages.length - 1
    ] as ToolMessage;
    return JSON.parse(lastMessage.content);
  }

  // #endregion nodes

  function callToolElse(elseStmt: (state: typeof AgentState.State) => string) {
    return function (state: typeof AgentState.State) {
      const messages = state.messages;
      const lastMessage = messages[messages.length - 1] as AIMessage;
      if (lastMessage?.tool_calls && lastMessage.tool_calls.length > 0) {
        return "CallTool";
      } else {
        return elseStmt(state);
      }
    };
  }

  const workflow = new StateGraph(AgentState)
    .addNode("PatientCreate", patientCreateNode)
    .addNode("PatientUpdate", patientUpdateNode)
    .addNode("DraftResourceApprove", draftResourceApproveNode)
    .addNode("ApproveResource", approveResourceNode)
    .addNode("CallTool", toolNode)
    .addNode("ToolMessageUpdateState", toolMessageUpdateStateNode);

  workflow.addConditionalEdges(
    "PatientCreate",
    callToolElse(() => "PatientCreate")
  );

  workflow.addConditionalEdges(
    "PatientUpdate",
    callToolElse(() => "PatientUpdate")
  );

  workflow.addConditionalEdges(
    "DraftResourceApprove",
    callToolElse(() => "DraftResourceApprove")
  );

  workflow.addEdge("CallTool", "ToolMessageUpdateState");

  workflow.addConditionalEdges("ToolMessageUpdateState", (state) => {
    if (state.sender === "Patient creater") return "DraftResourceApprove";
    if (state.sender === "Patient updater") return "DraftResourceApprove";
    if (state.sender === "Draft resource approver") {
      if (state.resourceApproved) return "ApproveResource";
      else return state.patientCreate ? "PatientCreate" : "PatientUpdate";
    }
    return "PatientCreate";
  });

  workflow.addEdge(START, "PatientCreate");

  const checkpointer = new MemorySaver();
  const graph = workflow.compile({ checkpointer });

  return graph;
}
