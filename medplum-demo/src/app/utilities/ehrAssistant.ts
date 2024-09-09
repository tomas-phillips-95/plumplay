import { BaseMessage } from "@langchain/core/messages";
import { Runnable } from "@langchain/core/runnables";
import { tool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { patientCreateSchema } from "./medplumZod";

type PatientCreate = z.infer<typeof patientCreateSchema>;

type BaseState = {
  messages: BaseMessage[];
};

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

abstract class Agent<StateT extends BaseState> {
  private runnable: Runnable;

  constructor(public name: string, llm: ChatOpenAI, tools: Tool<StateT>[]) {
    this.runnable = llm.bindTools(tools);
  }

  async run(state: StateT): Promise<StateT> {
    const result = await this.runnable.invoke(state);
    return await this.resultHandler(result);
  }

  protected abstract resultHandler(state: StateT): Promise<StateT>;
}

class Tool<_StateT extends BaseState> {
  constructor(private tool_: typeof tool) {}

  static from<T extends z.ZodTypeAny, StateT extends BaseState>(
    func: (result: z.infer<T>) => StateT,
    params: {
      name: string;
      description: string;
      schema: T;
    }
  ) {
    return new Tool<StateT>(tool(func, params) as any);
  }
}

class PatientCreateAgent extends Agent<State> {
  protected async resultHandler(state: State): Promise<State> {
    return state;
  }
}
