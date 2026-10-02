from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
def get_weather(city:str)->str:
    """Get the current weather for the given city"""
    return f"The weather in {city} is always clear and pleasant."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")