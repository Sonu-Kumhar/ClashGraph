import { HumanMessage } from "@langchain/core/messages";
import { StateSchema, MessagesValue, ReducedValue, type CompiledStateGraph, StateGraph, START, END } from "@langchain/langgraph";
import type { GraphNode } from "@langchain/langgraph";
import { z } from "zod";
import { createAgent, providerStrategy } from "langchain";
import { mistralModel, cohereModel, geminiModel } from "./models.ai.js";


const state = new StateSchema({
    problem: z.string().default(""),
    solution_1: z.string().default(""),
    solution_2: z.string().default(""),

    judge: z.object({
        solution_1_score: z.number().default(0),
        solution_2_score: z.number().default(0),
        solution_1_reasoning: z.string().default(""),
        solution_2_reasoning: z.string().default("")
    })

})


const solutionNode: GraphNode<typeof state> = async (state) =>{

    const [mistralResponse, cohereResponse] = await Promise.all([
        mistralModel.invoke(state.problem),
        cohereModel.invoke(state.problem)
    ])

    return {
        solution_1: mistralResponse.text,
        solution_2: cohereResponse.text
    }
}


const judgeNode: GraphNode<typeof state> = async (state) =>{
    const {problem, solution_1, solution_2} = state;

    /**  //for getting the structured output we will create Agent
     * judge response format:{
     * solution_1_score: 7,
     * solution_2_score: 8,
     * solution_1_reasoning: "reasoning for solution 1",
     * solution_2_reasoning: "reasoning for solution 2"
     * }
     */

    const judge = createAgent({
        model: geminiModel,
        responseFormat: providerStrategy(z.object({
            solution_1_score: z.number().min(0).max(10),
            solution_2_score: z.number().min(0).max(10),
            solution_1_reasoning: z.string(),
            solution_2_reasoning: z.string()
        })),
        systemPrompt: `You are a judge tasked with evaluating two solutions to the following problem: ${problem}. Please provide a score between 0 and 10 for each solution, along with your reasoning for the scores.`
    })

    const judgeResponse = await judge.invoke({
        messages: [
            new HumanMessage(`
                Problem: ${problem}
                Solution 1: ${solution_1}
                Solution 2: ${solution_2}
                please evaluate the two solutions and provide scores and reasoning.
                `)
        ]
    })

    const {
        solution_1_score,
        solution_2_score,
        solution_1_reasoning,
        solution_2_reasoning
    } = judgeResponse.structuredResponse

    return {
        judge: {
            solution_1_score,
            solution_2_score,
            solution_1_reasoning,
            solution_2_reasoning
        }
    }
}


const graph = new StateGraph(state)
    .addNode("solution", solutionNode)
    .addNode("judge_node",judgeNode)
    .addEdge(START, "solution")
    .addEdge("solution", "judge_node")
    .addEdge("judge_node", END)
    .compile()

export default async function runGraph(problem: string) {
    const result = await graph.invoke({
        problem: problem
    })

    return result;
}