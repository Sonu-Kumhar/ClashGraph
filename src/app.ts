import express from 'express';
import runGraph from "./ai/graph.ai.js"

const app = express();

app.get("/", async (req, res)=>{
    const result = await runGraph("What is the capital of France?");
    res.json(result);
});


app.get("/health", (req, res)=>{
    res.status(200).json({status : "ok"})
})



export default app;