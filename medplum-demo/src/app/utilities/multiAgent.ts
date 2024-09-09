import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import type { RunnableConfig } from "@langchain/core/runnables";
import { Runnable } from "@langchain/core/runnables";
import { StructuredTool, tool } from "@langchain/core/tools";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import {
  Annotation,
  NodeInterrupt,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
// #region schemas

enum ResourceType {
  Patient = "Patient",
}

enum Agent {
  PatientCreate = "patient-create",
  PatientUpdate = "patient-update",
  ResourceApproval = "resource-approval",
  ResourceCreate = "resource-create",
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
      "You are a helpful AI assistant, collaborating with other assistants." +
        " Use the provided tools to progress towards answering the question." +
        " If you are unable to fully answer, that's OK, another assistant with different tools " +
        " will help where you left off. Execute what you can to make progress." +
        " If you or any of the other assistants have the final answer or deliverable," +
        " prefix your response with FINAL ANSWER so the team knows to stop." +
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

const llm = new ChatOpenAI({ modelName: "gpt-4o" });

const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  sender: Annotation<string>({
    reducer: (x, y) => y ?? x ?? "user",
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

const resourceCreateTool = tool(
  () => {
    return {
      messages: new AIMessage({
        name: Agent.ResourceCreate,
        content: "Resource created/updated.",
      }),
      resourceApproved: false,
      patientCreate: null,
      patientUpdate: null,
    };
  },
  {
    name: "resource_create",
    description: "Creates a new resource in the EHR system.",
    schema: z.literal("create"),
  }
);

// #endregion tools

// #region agents

const patientCreateAgent = createAgent({
  llm,
  tools: [patientCreateTool],
  systemMessage:
    "You are an EHR assistant that helps create new patients in the system.",
});

const patientUpdateAgent = createAgent({
  llm,
  tools: [patientUpdateTool],
  systemMessage:
    "You are an EHR assistant that helps update existing patients in the system.",
});

const resourceApprovalAgent = createAgent({
  llm,
  tools: [resourceApprovalTool],
  systemMessage:
    "You are an assistant that verifies with the user the input for a new resource.",
});

const resourceCreateAgent = createAgent({
  llm,
  tools: [],
  systemMessage:
    "You are an assistant that helps create a new resource in the system.",
});

// #endregion agents

// #region nodes

async function patientCreateNode(
  state: typeof AgentState.State,
  config?: RunnableConfig
) {
  runAgentNode({
    state,
    agent: await patientCreateAgent,
    name: Agent.PatientCreate,
    config,
  });
}

async function patientUpdateNode(
  state: typeof AgentState.State,
  config?: RunnableConfig
) {
  runAgentNode({
    state,
    agent: await patientUpdateAgent,
    name: Agent.PatientUpdate,
    config,
  });
}

async function resourceApprovalNode(
  state: typeof AgentState.State,
  config?: RunnableConfig
) {
  runAgentNode({
    state,
    agent: await resourceApprovalAgent,
    name: Agent.ResourceApproval,
    config,
  });
}

async function resourceCreateNode(
  state: typeof AgentState.State,
  config?: RunnableConfig
) {
  runAgentNode({
    state,
    agent: await resourceCreateAgent,
    name: Agent.ResourceCreate,
    config,
  });
}

// #endregion nodes

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
  let result = await agent.invoke(state, config);
  if (!result?.tool_calls || result.tool_calls.length === 0) {
    result = new HumanMessage({ ...result, name: name });
  }
  return {
    messages: [result],
    sender: name,
  };
}

const tools = [patientCreateTool, patientUpdateTool, resourceApprovalTool];
const toolNode = new ToolNode<typeof AgentState.State>(tools);

function router(state: typeof AgentState.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  if (lastMessage?.tool_calls && lastMessage.tool_calls.length > 0) {
    return "call_tool";
  }
  return "continue";
}

const workflow = new StateGraph(AgentState)
  .addNode(Agent.PatientCreate, patientCreateNode)
  .addNode(Agent.PatientUpdate, patientUpdateNode)
  .addNode(Agent.ResourceApproval, resourceApprovalNode)
  .addNode(Agent.ResourceCreate, resourceCreateNode)
  .addNode("approve_resource", () => {
    throw new NodeInterrupt("Approval required");
  })
  .addNode("call_tool", toolNode);

workflow.addConditionalEdges(Agent.PatientCreate, router, {
  continue: Agent.ResourceApproval,
  call_tool: "call_tool",
});

workflow.addConditionalEdges(Agent.PatientUpdate, router, {
  continue: Agent.ResourceApproval,
  call_tool: "call_tool",
});

workflow.addConditionalEdges(Agent.ResourceApproval, (x) =>
  x.resourceApproved ? Agent.ResourceCreate : x.sender
);

workflow.addConditionalEdges(Agent.ResourceCreate, router, {
  continue: Agent.PatientUpdate,
  call_tool: "call_tool",
});

workflow.addConditionalEdges("call_tool", (x) => x.sender, {
  PatientUpdate: "approve_resource",
  PatientCreate: "approve_resource",
  ResourceApprove: Agent.ResourceApproval,
});

workflow.addEdge(START, Agent.PatientCreate);

const graph = workflow.compile();

async function invokeGraph() {
  const streamResults = await graph.stream(
    {
      messages: [
        new HumanMessage({
          content: "Generate a bar chart of the US gdp over the past 3 years.",
        }),
      ],
    },
    { recursionLimit: 150 }
  );

  const prettifyOutput = (output: Record<string, any>) => {
    const keys = Object.keys(output);
    const firstItem = output[keys[0]];

    if ("messages" in firstItem && Array.isArray(firstItem.messages)) {
      const lastMessage = firstItem.messages[firstItem.messages.length - 1];
      console.dir(
        {
          type: lastMessage._getType(),
          content: lastMessage.content,
          tool_calls: lastMessage.tool_calls,
        },
        { depth: null }
      );
    }

    if ("sender" in firstItem) {
      console.log({
        sender: firstItem.sender,
      });
    }
  };

  for await (const output of streamResults) {
    if (!output?.__end__) {
      prettifyOutput(output);
      console.log("----");
    }
  }
}
