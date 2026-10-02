from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

async def main():
    client = MultiServerMCPClient({
        "math" : {
            "command" : "uv",
            "args" : ["run", "python", "src/learnmcp/mathserver.py"],
            "transport" : "stdio"   
        },
        "weather" : {
            "url" : "http://localhost:8000/mcp",
            "transport" : "streamable-http"
        }
    })
    model = ChatGroq(model="openai/gpt-oss-20b")
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    math_result = await agent.ainvoke({"messages" : [{"role" : "user", "content" : "You MUST use the MCP math tools to answer this. Do not calculate it yourself.cWhat is 5 + 10 * 2?"}]})
    print(math_result["messages"][-1].content)
    print()
    weather_result = await agent.ainvoke({"messages" : [{"role" : "user", "content" : "You MUST use the MCP weather tool to answer this. Do not find it yourself. What is the current weather in Mumbai?"}]})
    print(weather_result["messages"][-1].content)
    print()

asyncio.run(main())