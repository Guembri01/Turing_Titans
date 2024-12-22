from typing import Any, Dict
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from openai import InternalServerError
import logging
import json
import re
import os
from pathlib import Path
from langchain.agents import AgentExecutor

# Type definitions
State = Dict[str, Any]

# Set up logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_message(message: dict, name: str) -> BaseMessage:
    """Create a BaseMessage object based on the message type."""
    content = message.get("content", "")
    message_type = message.get("type", "").lower()
    
    logger.debug(f"Creating message of type {message_type} for {name}")
    return HumanMessage(content=content) if message_type == "human" else AIMessage(content=content, name=name)

def _create_error_state(state: State, error_message: AIMessage, name: str, error_type: str) -> State:
    """Create an error state when an exception occurs."""
    logger.info(f"Creating error state for {name}: {error_type}")
    return {
        "messages": state.get("messages", []) + [error_message],
        "hypothesis": str(state.get("hypothesis", "")),
        "process": str(state.get("process", "")),
        "process_decision": str(state.get("process_decision", "")),
        "visualization_state": str(state.get("visualization_state", "")),
        "searcher_state": str(state.get("searcher_state", "")),
        "code_state": str(state.get("code_state", "")),
        "report_section": str(state.get("report_section", "")),
        "quality_review": str(state.get("quality_review", "")),
        "needs_revision": bool(state.get("needs_revision", False)),
        "sender": name
    }

def process_router(state: State) -> str:
    """Route to next step based on process decision."""
    logger.info("Entering process_router")
    
    try:
        process_decision = state.get("process_decision", "")
        
        # Handle the case where process_decision is an AIMessage
        if isinstance(process_decision, AIMessage):
            content = process_decision.content
        else:
            content = process_decision
            
        # Convert string representation of dictionary to actual dictionary
        if isinstance(content, str):
            # Remove any leading/trailing whitespace and quotes
            content = content.strip().strip("'").strip('"')
            if content.startswith("{") and content.endswith("}"):
                try:
                    # Use ast.literal_eval for safer dictionary string evaluation
                    import ast
                    decision_dict = ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    logger.warning("Failed to parse process decision string")
                    return "Process"
            else:
                return "Process"
        else:
            decision_dict = content
            
        # Validate the decision dictionary
        if isinstance(decision_dict, dict) and "next" in decision_dict:
            next_step = decision_dict.get("next")
            if next_step in ["Coder", "Searcher", "Visualizer", "Reporter", "Process"]:
                logger.info(f"Process router decision: {next_step}")
                return next_step
        
        logger.warning(f"Invalid next step in process decision: {decision_dict}")
        return "Process"
        
    except Exception as e:
        logger.error(f"Error in process router: {str(e)}")
        return "Process"

def agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """Process an agent's action and update the state accordingly."""
    logger.info(f"Processing agent: {name}")
    try:
        result = agent.invoke(state)
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)

        # For process_agent, handle the dictionary-like output
        if name == "process_agent":
            try:
                import ast
                # If output is already a dictionary, use it directly
                if isinstance(output, dict):
                    process_decision = output
                else:
                    # Clean and parse the string representation
                    cleaned_output = output.strip().strip("'").strip('"')
                    if cleaned_output.startswith("{") and cleaned_output.endswith("}"):
                        process_decision = ast.literal_eval(cleaned_output)
                    else:
                        process_decision = output

                state["process_decision"] = process_decision
                logger.info(f"Process decision updated: {process_decision}")
            except Exception as e:
                logger.warning(f"Could not parse process decision: {str(e)}")
                state["process_decision"] = output

        ai_message = AIMessage(content=output, name=name)
        state["messages"].append(ai_message)
        state["sender"] = name

        # Update other state components
        if name == "hypothesis_agent" and not state["hypothesis"]:
            state["hypothesis"] = ai_message
            logger.info("Hypothesis updated")
        elif name == "visualization_agent":
            state["visualization_state"] = ai_message
            logger.info("Visualization state updated")
        elif name == "searcher_agent":
            state["searcher_state"] = ai_message
            logger.info("Searcher state updated")
        elif name == "report_agent":
            state["report_section"] = ai_message
            logger.info("Report section updated")
        elif name == "quality_review_agent":
            state["quality_review"] = ai_message
            state["needs_revision"] = "revision needed" in str(output).lower()
            logger.info(f"Quality review updated. Needs revision: {state['needs_revision']}")

        logger.info(f"Agent {name} processing completed")
        return state

    except Exception as e:
        logger.error(f"Error in agent {name}: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return _create_error_state(state, error_message, name, "Agent processing error")

# Add a helper function to safely evaluate dictionary strings
def safe_eval_dict(s: str) -> dict:
    """Safely evaluate a string representation of a dictionary."""
    try:
        import ast
        return ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        logger.error(f"Error evaluating dictionary string: {str(e)}")
        return {}
def note_agent_node(state: State, agent: AgentExecutor, name: str) -> State:
    """Process the note agent's action and update the entire state."""
    logger.info(f"Processing note agent: {name}")
    try:
        current_messages = state.get("messages", [])
        
        # Handle message trimming for long conversations
        head_messages, tail_messages = [], []
        if len(current_messages) > 6:
            head_messages = current_messages[:2]
            tail_messages = current_messages[-2:]
            state = {**state, "messages": current_messages[2:-2]}
            logger.debug("Trimmed messages for processing")
        
        result = agent.invoke(state)
        output = result["output"] if isinstance(result, dict) and "output" in result else str(result)

        # Clean and parse output
        cleaned_output = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', output)
        try:
            parsed_output = json.loads(cleaned_output)
        except json.JSONDecodeError:
            logger.warning("Failed to parse output as JSON, using raw output")
            parsed_output = {"messages": [{"content": cleaned_output, "type": "ai"}]}

        # Create new messages
        new_messages = [create_message(msg, name) for msg in parsed_output.get("messages", [])]
        messages = new_messages if new_messages else current_messages
        combined_messages = head_messages + messages + tail_messages

        # Update state
        updated_state = {
            "messages": combined_messages,
            "hypothesis": str(parsed_output.get("hypothesis", state.get("hypothesis", ""))),
            "process": str(parsed_output.get("process", state.get("process", ""))),
            "process_decision": str(parsed_output.get("process_decision", state.get("process_decision", ""))),
            "visualization_state": str(parsed_output.get("visualization_state", state.get("visualization_state", ""))),
            "searcher_state": str(parsed_output.get("searcher_state", state.get("searcher_state", ""))),
            "code_state": str(parsed_output.get("code_state", state.get("code_state", ""))),
            "report_section": str(parsed_output.get("report_section", state.get("report_section", ""))),
            "quality_review": str(parsed_output.get("quality_review", state.get("quality_review", ""))),
            "needs_revision": bool(parsed_output.get("needs_revision", state.get("needs_revision", False))),
            "sender": name
        }
        
        logger.info("Note agent processing completed successfully")
        return updated_state

    except Exception as e:
        logger.error(f"Error in note agent: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Error: {str(e)}", name=name)
        return _create_error_state(state, error_message, name, "Note agent error")

def human_choice_node(state: State) -> State:
    """Handle human input to choose the next step in the process."""
    logger.info("Prompting for human choice")
    print("\nPlease choose the next step:")
    print("1. Regenerate hypothesis")
    print("2. Continue the research process")
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice in ["1", "2"]:
                break
            print("Invalid input. Please enter 1 or 2.")
        except Exception as e:
            logger.error(f"Error in human input: {str(e)}")
            print("An error occurred. Please try again.")

    if choice == "1":
        modification_areas = input("Specify areas to modify in the hypothesis: ").strip()
        content = f"Regenerate hypothesis. Areas to modify: {modification_areas}"
        state["hypothesis"] = ""
        state["modification_areas"] = modification_areas
        logger.info(f"Hypothesis regeneration requested for areas: {modification_areas}")
    else:
        content = "Continue the research process"
        state["process"] = "Continue"
        logger.info("Continuing research process")
    
    state["messages"].append(HumanMessage(content=content))
    state["sender"] = "human"
    return state

def human_review_node(state: State) -> State:
    """Display current state and get human feedback."""
    logger.info("Starting human review")
    try:
        print("\nCurrent Research Progress:")
        print("---------------------------")
        for key, value in state.items():
            if key != "messages":
                print(f"{key}: {value}")
        
        print("\nDo you need additional analysis or modifications?")
        while True:
            try:
                choice = input("Enter 'yes' to continue analysis, or 'no' to end research: ").lower().strip()
                if choice in ['yes', 'no']:
                    break
                print("Invalid input. Please enter 'yes' or 'no'.")
            except Exception as e:
                logger.error(f"Error in input: {str(e)}")
                print("An error occurred. Please try again.")

        if choice == 'yes':
            while True:
                try:
                    request = input("Enter your additional analysis request: ").strip()
                    if request:
                        state["messages"].append(HumanMessage(content=request))
                        state["needs_revision"] = True
                        break
                    print("Request cannot be empty. Please try again.")
                except Exception as e:
                    logger.error(f"Error in request input: {str(e)}")
                    print("An error occurred. Please try again.")
        else:
            state["needs_revision"] = False
        
        state["sender"] = "human"
        logger.info("Human review completed successfully")
        return state

    except Exception as e:
        logger.error(f"Error in human review: {str(e)}", exc_info=True)
        return state

def refiner_node(state: State, agent: AgentExecutor, name: str) -> State:
    """Process and refine research materials."""
    logger.info("Starting refiner node processing")
    try:
        storage_path = Path(os.getenv('STORAGE_PATH', './data_storage/'))
        
        # Collect and process materials
        materials = []
        try:
            # Process MD files
            for md_file in storage_path.glob("*.md"):
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()
                        materials.append(f"MD file '{md_file.name}':\n{content}")
                except Exception as e:
                    logger.error(f"Error reading MD file {md_file}: {str(e)}")
                    continue

            # Process PNG files
            png_files = list(storage_path.glob("*.png"))
            materials.extend(f"PNG file: '{png_file.name}'" for png_file in png_files)
            
        except Exception as e:
            logger.error(f"Error collecting materials: {str(e)}")
            materials.append("Error collecting some materials")

        # Create refiner state with materials
        refiner_state = state.copy()
        combined_materials = "\n\n".join(materials)
        refiner_state["messages"] = [HumanMessage(content=f"Report materials:\n{combined_materials}")]

        try:
            # Process with full content
            result = agent.invoke(refiner_state)
            output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
            
            # Update state with results
            state["messages"].append(AIMessage(content=output, name=name))
            
            # Try to parse any process decisions
            if isinstance(output, dict) and "process_decision" in output:
                try:
                    process_decision = json.loads(output["process_decision"])
                    state["process_decision"] = process_decision
                except json.JSONDecodeError:
                    logger.warning("Failed to parse process decision as JSON")
                    state["process_decision"] = output.get("process_decision", "")
            
        except Exception as token_error:
            logger.warning(f"Full content processing failed: {str(token_error)}")
            
            # Fallback to simplified content
            md_files = [f"MD file: '{f.name}'" for f in storage_path.glob("*.md")]
            png_files = [f"PNG file: '{f.name}'" for f in storage_path.glob("*.png")]
            simplified_materials = "\n".join(md_files + png_files)
            
            refiner_state["messages"] = [HumanMessage(content=f"Report materials (files only):\n{simplified_materials}")]
            
            try:
                result = agent.invoke(refiner_state)
                output = result["output"] if isinstance(result, dict) and "output" in result else str(result)
                state["messages"].append(AIMessage(content=output, name=name))
            except Exception as e:
                logger.error(f"Simplified content processing failed: {str(e)}")
                raise

        state["sender"] = name
        logger.info("Refiner node processing completed successfully")
        return state

    except Exception as e:
        logger.error(f"Error in refiner node: {str(e)}", exc_info=True)
        error_message = AIMessage(content=f"Refiner process error: {str(e)}", name=name)
        return _create_error_state(state, error_message, name, "Refiner error")

def initialize_state() -> State:
    """Initialize the research state with default values."""
    return {
        "messages": [],
        "hypothesis": "",
        "process": "",
        "process_decision": "",
        "visualization_state": "",
        "searcher_state": "",
        "code_state": "",
        "report_section": "",
        "quality_review": "",
        "needs_revision": False,
        "sender": None
    }