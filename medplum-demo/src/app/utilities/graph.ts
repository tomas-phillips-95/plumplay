import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import { Runnable } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { patientCreateSchema } from "./medplumZod";

const Node = {
  Agent: "agent",
  Tools: "tools",
  Start: START,
  End: END,
} as const;

type PatientCreate = z.infer<typeof patientCreateSchema>;

type GraphNode = (typeof Node)[keyof typeof Node];

type State =
  | {
      messages: BaseMessage[];
    }
  | {
      patientCreate: PatientCreate;
      messages: BaseMessage[];
    }
  | {
      patient: PatientCreate;
      messages: BaseMessage[];
    };

/**
 * - patient create
 * - patient get
 * - patient update
 */

class EhrAssistant {
  private model: ChatOpenAI;

  constructor(private apiKey: string) {
    this.model = new ChatOpenAI({
      model: "gpt-4o",
      apiKey: apiKey,
    });

    const StateAnnotation = Annotation.Root({
      messages: Annotation<BaseMessage[]>({
        reducer: (x, y) => x.concat(y),
      }),
      patientCreate: Annotation<PatientCreate | undefined>({
        reducer: (_, y) => y,
        default: () => undefined,
      }),
      patient: Annotation<PatientCreate | undefined>({
        reducer: (_, y) => y,
        default: () => undefined,
      }),
    });

    const tools = [EhrAssistant.createPatientTool];
    const toolNode = new ToolNode(tools);
    this.model.bindTools(tools.map(convertToOpenAITool));

    const workflow = new StateGraph(StateAnnotation)
      .addNode(Node.Agent, this.callModel)
      .addNode(Node.Tools, toolNode)
      .addEdge(START, Node.Agent)
      .addConditionalEdges(Node.Agent, this.shouldContinue);
  }

  static createPatientTool = tool(
    async (patient) => {
      console.log(patient);
    },
    {
      name: "createPatient",
      description: "Call to create a new patient in the EHR",
      schema: patientCreateSchema,
    }
  );

  shouldContinue(state: State): GraphNode | typeof END {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    if (lastMessage.tool_calls?.length) {
      return GraphNode.Tools;
    }
    return END;
  }

  async callModel(state: State) {
    const messages = state.messages;
    const response = await this.model.invoke(messages);
    return { messages: [response] };
  }
}

abstract class Agent<S> {
  constructor(public name: string, private agent: Runnable) {}

  async runAgentNode<S>(state: S) {
    let result = await this.agent.invoke(state);
    if (!result?.tool_calls || result.tool_calls.length === 0) {
      result = new HumanMessage({ ...result, name: this.name });
    }
    return {
      messages: [result],
      sender: this.name,
    };
  }

  abstract handle(state: S): Promise<S>;
}
