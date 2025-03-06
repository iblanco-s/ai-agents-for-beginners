import os
import json
import time
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import matplotlib.pyplot as plt
from diskcache import Cache

# Initialize cache
cache = Cache("./response_cache")

# Initialize evaluation metrics
evaluation_metrics = {
    "total_requests": 0,
    "cached_responses": 0,
    "router_decisions": {
        "simple_model": 0,
        "complex_model": 0
    },
    "execution_times": [],
    "successful_tasks": 0,
    "failed_tasks": 0,
    "tool_usage": {},
    "user_feedback_scores": []
}

# Initialize clients for different model sizes
simple_client = AzureOpenAI(
    api_key=os.environ["GITHUB_TOKEN"],  
    api_version="2023-05-15",
    azure_endpoint="https://models.inference.ai.azure.com",
)

complex_client = AzureOpenAI(
    api_key=os.environ["GITHUB_TOKEN"],
    api_version="2023-05-15",
    azure_endpoint="https://models.inference.ai.azure.com",
)

# Define available tools
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    # In a real system, this would call a weather API
    return f"The weather in {location} is sunny and 75°F."

def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real system, this would call a search API
    return f"Search results for '{query}': Found 10 relevant articles."

def calculate_expression(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

# Define tools for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_expression",
            "description": "Calculate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to calculate"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# Function to handle tool calls
def handle_tool_call(tool_call):
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)
    
    # Track tool usage for evaluation
    if function_name in evaluation_metrics["tool_usage"]:
        evaluation_metrics["tool_usage"][function_name] += 1
    else:
        evaluation_metrics["tool_usage"][function_name] = 1
    
    # Call the appropriate function
    if function_name == "get_weather":
        return get_weather(arguments["location"])
    elif function_name == "search_web":
        return search_web(arguments["query"])
    elif function_name == "calculate_expression":
        return calculate_expression(arguments["expression"])
    else:
        return f"Unknown function: {function_name}"

# Router function to determine which model to use based on task complexity
def route_to_appropriate_model(user_query: str) -> AzureOpenAI:
    """
    Routes the query to the appropriate model based on complexity.
    
    Simple model criteria:
    - Short queries (less than 100 characters)
    - Basic calculations or information retrieval
    - No complex reasoning required
    
    Complex model for everything else.
    """
    # Simple heuristics for routing
    is_simple = (
        len(user_query) < 100 and
        not any(complex_term in user_query.lower() for complex_term in 
                ["explain", "analyze", "compare", "evaluate", "why", "how would", "design"])
    )
    
    if is_simple:
        evaluation_metrics["router_decisions"]["simple_model"] += 1
        return simple_client
    else:
        evaluation_metrics["router_decisions"]["complex_model"] += 1
        return complex_client

# Function to create a cache key from user query
def create_cache_key(user_query: str) -> str:
    """Create a hash of the user query to use as a cache key."""
    return hashlib.md5(user_query.encode()).hexdigest()

# Main function to process user requests
def process_user_request(user_query: str, max_turns: int = 5) -> Dict:
    """
    Process a user request using the AI agent system.
    
    Args:
        user_query: The user's question or request
        max_turns: Maximum number of turns to prevent infinite loops
        
    Returns:
        Dict containing the final response and metrics
    """
    start_time = time.time()
    evaluation_metrics["total_requests"] += 1
    
    # Check cache first
    cache_key = create_cache_key(user_query)
    cached_response = cache.get(cache_key)
    
    if cached_response:
        evaluation_metrics["cached_responses"] += 1
        execution_time = time.time() - start_time
        evaluation_metrics["execution_times"].append(execution_time)
        
        return {
            "response": cached_response,
            "source": "cache",
            "execution_time": execution_time
        }
    
    # Route to appropriate model
    client = route_to_appropriate_model(user_query)
    
    # Initialize conversation
    messages = [{"role": "system", "content": """You are a helpful AI assistant. 
    Follow these guidelines:
    1. Use the available tools when appropriate
    2. Provide concise and accurate responses
    3. If you don't know the answer, say so
    4. Stop when the task is complete
    5. Don't hallucinate information
    """}]
    
    messages.append({"role": "user", "content": user_query})
    
    turn_count = 0
    final_response = ""
    
    # Conversation loop with turn limit to prevent infinite loops
    while turn_count < max_turns:
        turn_count += 1
        
        # Get response from model
        try:
            response = client.chat.completions.create(
                model=client.model if hasattr(client, 'model') else ("gpt-4o-mini" if client == simple_client else "gpt-4o"),
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            assistant_message = response.choices[0].message
            messages.append({"role": assistant_message.role, "content": assistant_message.content})
            
            # Check if the model wants to use a tool
            if assistant_message.tool_calls:
                for tool_call in assistant_message.tool_calls:
                    tool_response = handle_tool_call(tool_call)
                    
                    # Add the tool response to the messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                
                # Continue the conversation after tool use
                continue
            
            # If no tool calls, this is the final response
            final_response = assistant_message.content
            evaluation_metrics["successful_tasks"] += 1
            break
            
        except Exception as e:
            final_response = f"Error: {str(e)}"
            evaluation_metrics["failed_tasks"] += 1
            break
    
    # Handle case where max turns were reached
    if turn_count >= max_turns and not final_response:
        final_response = "I apologize, but I was unable to complete your request within the maximum number of steps."
        evaluation_metrics["failed_tasks"] += 1
    
    # Cache the response
    cache.set(cache_key, final_response)
    
    # Record execution time
    execution_time = time.time() - start_time
    evaluation_metrics["execution_times"].append(execution_time)
    
    return {
        "response": final_response,
        "source": "model",
        "model_used": "gpt-4o-mini" if client == simple_client else "gpt-4o",
        "turns": turn_count,
        "execution_time": execution_time
    }

# Function to collect user feedback
def collect_user_feedback(response: Dict, score: int) -> None:
    """
    Collect user feedback on the response quality.
    
    Args:
        response: The response dictionary
        score: User feedback score (1-5)
    """
    evaluation_metrics["user_feedback_scores"].append({
        "score": score,
        "model": response.get("model_used", "cache"),
        "execution_time": response["execution_time"],
        "turns": response.get("turns", 0),
        "timestamp": datetime.now().isoformat()
    })

# Function to generate evaluation reports
def generate_evaluation_report() -> Dict:
    """Generate a comprehensive evaluation report of the AI agent system."""
    
    # Calculate average metrics
    avg_execution_time = sum(evaluation_metrics["execution_times"]) / max(len(evaluation_metrics["execution_times"]), 1)
    cache_hit_rate = evaluation_metrics["cached_responses"] / max(evaluation_metrics["total_requests"], 1) * 100
    success_rate = evaluation_metrics["successful_tasks"] / max(evaluation_metrics["total_requests"], 1) * 100
    
    # Calculate average feedback score if available
    avg_feedback_score = 0
    if evaluation_metrics["user_feedback_scores"]:
        avg_feedback_score = sum(item["score"] for item in evaluation_metrics["user_feedback_scores"]) / len(evaluation_metrics["user_feedback_scores"])
    
    report = {
        "summary": {
            "total_requests": evaluation_metrics["total_requests"],
            "cache_hit_rate": cache_hit_rate,
            "success_rate": success_rate,
            "avg_execution_time_seconds": avg_execution_time,
            "avg_user_feedback": avg_feedback_score
        },
        "model_usage": {
            "simple_model": evaluation_metrics["router_decisions"]["simple_model"],
            "complex_model": evaluation_metrics["router_decisions"]["complex_model"]
        },
        "tool_usage": evaluation_metrics["tool_usage"],
        "detailed_feedback": evaluation_metrics["user_feedback_scores"]
    }
    
    return report

# Function to visualize evaluation metrics
def visualize_metrics():
    """Generate visualizations of key metrics for monitoring."""
    
    # Create a figure with multiple subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Model usage
    plt.subplot(2, 2, 1)
    model_usage = [
        evaluation_metrics["router_decisions"]["simple_model"],
        evaluation_metrics["router_decisions"]["complex_model"],
        evaluation_metrics["cached_responses"]
    ]
    plt.pie(model_usage, labels=['Simple Model', 'Complex Model', 'Cache'], autopct='%1.1f%%')
    plt.title('Resource Usage Distribution')
    
    # Plot 2: Success vs Failure
    plt.subplot(2, 2, 2)
    plt.bar(['Success', 'Failure'], 
            [evaluation_metrics["successful_tasks"], evaluation_metrics["failed_tasks"]])
    plt.title('Task Completion Status')
    plt.ylabel('Number of Tasks')
    
    # Plot 3: Tool usage
    plt.subplot(2, 2, 3)
    tool_names = list(evaluation_metrics["tool_usage"].keys())
    tool_counts = list(evaluation_metrics["tool_usage"].values())
    plt.bar(tool_names, tool_counts)
    plt.title('Tool Usage')
    plt.ylabel('Number of Uses')
    plt.xticks(rotation=45)
    
    # Plot 4: Execution times histogram
    plt.subplot(2, 2, 4)
    plt.hist(evaluation_metrics["execution_times"], bins=10)
    plt.title('Response Time Distribution')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('agent_metrics.png')
    plt.close()

# Example usage of the system
def demo_ai_agent_system():
    """Demonstrate the AI agent system with example queries."""
    
    example_queries = [
        "What's the weather like in Seattle?",
        "Can you help me understand quantum computing?",
        "Calculate 125 * 37",
        "What's the weather like in Seattle?",  # Repeated to show caching
        "Design a marketing strategy for a new startup in the renewable energy sector",
        "What's 2+2?",
        "Explain the difference between deep learning and machine learning",
        "Search for information about climate change"
    ]
    
    results = []
    
    print("Running example queries through the AI agent system...")
    for query in example_queries:
        print(f"\nProcessing query: '{query}'")
        result = process_user_request(query)
        
        # Simulate user feedback (random score between 1-5)
        import random
        feedback_score = random.randint(1, 5)
        collect_user_feedback(result, feedback_score)
        
        print(f"Response: {result['response'][:100]}..." if len(result['response']) > 100 else f"Response: {result['response']}")
        print(f"Source: {result['source']}")
        if result['source'] == 'model':
            print(f"Model used: {result['model_used']}")
            print(f"Turns: {result['turns']}")
        print(f"Execution time: {result['execution_time']:.2f} seconds")
        
        results.append(result)
    
    # Generate and display evaluation report
    print("\n\n===== EVALUATION REPORT =====")
    report = generate_evaluation_report()
    print(json.dumps(report, indent=2))
    
    # Visualize metrics
    visualize_metrics()
    print("\nMetrics visualization saved as 'agent_metrics.png'")
    
    return results

# Main execution
if __name__ == "__main__":
    print("AI Agent Production System with Cost Management and Evaluation")
    print("=" * 80)
    demo_results = demo_ai_agent_system()
    
    # Save evaluation data for future analysis
    pd.DataFrame(evaluation_metrics["user_feedback_scores"]).to_csv("feedback_data.csv", index=False)
    
    print("\nSystem execution complete.")
