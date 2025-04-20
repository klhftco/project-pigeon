# app.py
from flask import Flask, render_template, request, jsonify, send_file
from openai import OpenAI
import json
import os

app = Flask(__name__)

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "locate_person",
            "description": "Command the drone to locate a person based on visual description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "point_of_interest": {
                        "type": "string",
                        "description": "Point of interest for the drone to focus on. (Like a yellow hat)"
                    },

                },
                "required": ["point_of_interest"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_distance",
            "description": "Adjust drone's distance to the object (move closer or away).",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["closer", "away"],
                        "description": "Direction to move relative to the object."
                    },
                },
                "required": ["direction"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "spin_around",
            "description": "Make the drone perform a 360° spin around the current object of interest.",
            "parameters": {
                "type": "object",
                "properties": {}, 
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_drone_image",
            "description": "Get the current image from the drone camera.",
            "parameters": {
                "type": "object",
                "properties": {}, 
                "additionalProperties": False
            }
        }
    }
]

conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    
    conversation_history.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=conversation_history,
        tools=tools,
        tool_choice="auto"
    )
    
    # Get the assistant's message
    assistant_message = response.choices[0].message
    
    # Add the assistant message to conversation history
    conversation_history.append(assistant_message.model_dump())
    
    # Process tool calls if any
    action_results = []
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            function_id = tool_call.id
            arguments = json.loads(tool_call.function.arguments)
            
            # Execute the function call (simulated)
            result = execute_function(function_name, arguments)
            action_results.append(result)
            
            # Add tool response to conversation history
            conversation_history.append({
                "role": "tool",
                "tool_call_id": function_id,
                "content": json.dumps(result)
            })
        
        # Get a follow-up response after tool use
        follow_up_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto"
        )
        
        # Update assistant message with the follow-up response
        assistant_message = follow_up_response.choices[0].message
        conversation_history.append(assistant_message.model_dump())
        
        return jsonify({
            "assistant_message": "",  # Empty string to avoid duplicate messages
            "action_results": action_results
        })
    
    # Return results for non-tool messages
    return jsonify({
        "assistant_message": assistant_message.content,
        "action_results": action_results
    })

def execute_function(function_name, arguments):
    """Simulate executing drone functions and return results."""
    if function_name == "locate_person":
        poi = arguments.get('point_of_interest', 'unknown')
        return {
            "action": "locate_person",
            "message": f"Drone is scanning for {poi}."  # Use the variable 'poi'
        }
    elif function_name == "adjust_distance":
        return {
            "action": "adjust_distance",
            "message": f"Drone is moving {arguments['direction']}."
        }
    elif function_name == "spin_around":
        return {
            "action": "spin_around",
            "message": f"Drone is performing a 360° spin around the object."
        }
    elif function_name == "get_drone_image":
        return {
            "action": "get_drone_image",
            "message": "Drone image captured successfully.",
            "image": "image_sample.jpg"
        }
    else:
        return {
            "action": "unknown",
            "message": f"Unknown function: {function_name}"
        }

@app.route('/drone_image/<filename>')
def drone_image(filename):
    """Serve drone images from the project root directory."""
    # Use the direct path to the image at the root level
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    # Check if file exists and print path for debugging
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        # Fallback to looking in static folder if exists
        static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', filename)
        if os.path.exists(static_path):
            return send_file(static_path, mimetype='image/jpeg')
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)