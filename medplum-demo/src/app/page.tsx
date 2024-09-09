"use client";

import { HumanMessage } from "@langchain/core/messages";
import { MedplumClient } from "@medplum/core";
import { useEffect, useState } from "react";
import { createGraph } from "./utilities/new";

const MEDPLUM_CLIENT_ID = "7dd275ab-c8a8-4e19-a853-5a6df878366d";
const MEDPLUM_CLIENT_SECRET =
  "5b4dc1addac4ae41265407b7f17131d4ffdb03fd4699bd6ba5475e1d4d02442d";

const MEDPLUM_SERVER_BASE_URL = "http://localhost:8103";
const OPENAI_API_KEY = "";

const medplum = new MedplumClient({
  clientId: MEDPLUM_CLIENT_ID,
  clientSecret: MEDPLUM_CLIENT_SECRET,
  baseUrl: MEDPLUM_SERVER_BASE_URL,
});

export default function Home() {
  const [userContent, setUserContent] = useState<string>("");
  const [responses, setResponses] = useState<string[]>([]);
  const [graph, setGraph] = useState<Awaited<
    ReturnType<typeof createGraph>
  > | null>(null);
  const [hasInvokedGraph, setHasInvokedGraph] = useState<boolean>(false);

  useEffect(() => {
    (async () => {
      const graph = await createGraph(OPENAI_API_KEY);
      setGraph(graph);
    })();
  }, []);

  const handleSend = async () => {
    if (!graph) return;

    let result: any;
    if (hasInvokedGraph) {
      const state = await graph.getState({
        configurable: { thread_id: "asasdfd" },
      });
      await graph.updateState(
        {
          configurable: { thread_id: "asasdfd" },
          recursionLimit: 5,
        },
        {
          ...state,
          messages: [new HumanMessage(userContent)],
        }
      );
      result = await graph.invoke(null, {
        configurable: { thread_id: "asasdfd" },
        recursionLimit: 5,
        debug: true,
      });
    } else {
      result = await graph.invoke(
        { messages: [new HumanMessage(userContent)] },
        {
          configurable: { thread_id: "asasdfd" },
          recursionLimit: 5,
          debug: true,
        }
      );
      setHasInvokedGraph(true);
    }
    const lastMessage = result.messages[result.messages.length - 1];

    setResponses([...responses, lastMessage.content]);
    setUserContent("");
  };

  return (
    <main className="absolute w-full h-full p-8 bg-zinc-200 overflow-hidden">
      <div className="flex flex-row gap-4 h-full w-full">
        <div className="flex flex-1 flex-col h-full">
          <textarea
            className="w-full flex-1 bg-zinc-50 resize-none rounded-lg text-zinc-900 p-4 mb-4 h-full"
            value={userContent}
            onChange={({ target: { value } }) => setUserContent(value)}
          />
          <button
            className="bg-zinc-500 text-zinc-50 p-2 rounded-lg"
            disabled={!graph}
            onClick={handleSend}
          >
            Send
          </button>
        </div>
        <div className="flex flex-1 h-full flex-col">
          <textarea
            className="w-full h-full bg-zinc-50 resize-none rounded-lg text-zinc-900 p-4"
            value={responses.join("\n")}
            disabled
          />
        </div>
      </div>
    </main>
  );
}
