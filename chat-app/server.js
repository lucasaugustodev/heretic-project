const express = require("express");
const path = require("path");
const http = require("http");

const app = express();
app.use(express.json({ limit: "10mb" }));
app.use(express.static(path.join(__dirname, "public")));

const API_URL = "http://localhost:8000/v1/chat/completions";

app.post("/api/chat", async (req, res) => {
  const { messages, temperature = 0.7, max_tokens = 2048 } = req.body;

  try {
    const body = JSON.stringify({
      model: "heretic-mistral-7b",
      messages,
      temperature,
      max_tokens,
      stream: true,
    });

    const options = {
      hostname: "localhost",
      port: 8000,
      path: "/v1/chat/completions",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: "Bearer sk-local-heretic",
        "Content-Length": Buffer.byteLength(body),
      },
    };

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    const proxyReq = http.request(options, (proxyRes) => {
      proxyRes.on("data", (chunk) => res.write(chunk));
      proxyRes.on("end", () => res.end());
    });

    proxyReq.on("error", (err) => {
      console.error("API error:", err.message);
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    });

    proxyReq.write(body);
    proxyReq.end();
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

const PORT = 3333;
app.listen(PORT, () => {
  console.log(`Heretic Chat running at http://localhost:${PORT}`);
});
