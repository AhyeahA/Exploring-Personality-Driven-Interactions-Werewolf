# Exploring-Personality-Driven-Interactions-Werewolf

## Overview

This project aims to explore the language dynamics and styles in Large Language Model (LLM) agents. The primary objective is to explore how assigning different personality profiles shapes the language used by LLM agents during their interactions, specifically within the demanding environment of communication games.


## The Werewolf Game Testbed and Implementation 

We utilize the Werewolf game as a central testbed for this research. The code for this project is built upon the framework proposed in the paper ["Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf."](https://arxiv.org/abs/2309.04658). The original GitHub repository providing the code is available [here](https://github.com/xuyuzhuang11/Werewolf). 

## Presentation

For a comprehensive overview of the project, its objectives, and proposed research plan, please refer to the presentation available [here](https://docs.google.com/presentation/d/1UsMIvsHWXwR9H4MEkN_nHSJYk0WjpJqD/edit?usp=sharing&ouid=110113965104854581749&rtpof=true&sd=true). 

## Introduction

The research area of Artificial Intelligence in social interaction, particularly within the context of communicative games and deception, is a rapidly growing field. Understanding how AI agents can navigate complex social dynamics, including cooperation and deception, is crucial for developing more sophisticated and human-like interactions. This is important for various applications, ranging from creating believable virtual agents with distinct personalities to building robust systems that can detect and mitigate deceptive information. By studying AI agents in social deduction games like Werewolf and Mafia, we can gain insights into emergent strategic behaviors, the role of communication, and the subtle cues that indicate truthfulness or deception.

This research aims to investigate the capabilities of AI agents, particularly those powered by Large Language Models (LLMs), in engaging in strategic play and exhibiting behaviors related to personality and deception within communicative games. We seek to understand how different design choices, such as incorporating personality cues within the realm of the five-factor model, will influence the behavior and believability of these agents. Furthermore, we aim to explore methods for analyzing linguistic patterns exhibited by both AI and human players in these interactive settings.

## Research Questions

To achieve this, we will address the following research questions:
*	Can LLMs exhibit consistent personality traits through language?
*	Do these traits influence communication patterns of other agents like they do in humans?

## Exploratory Plan

Through prompt engineering, we aim to explore how prompts can be adjusted to effectively inject personality traits into large language models (LLMs), and to identify additional factors necessary to achieve this.

## Main Manipulation

The Werewolf agents will be designed to exhibit high neuroticism, while all other agents will remain constant and neutral.

# Prompt Example

## Main prompt

The main prompt, which specifies the rules and objectives of the game, has been adjusted so that agents are aware that some of them may be assigned a personality and should be mindful of how to act accordingly.

This prompt is implemented in the **`Werewolf-main\examples\werewolf.json`**

```json
{
    "name": "Werewolf",
    "global_prompt": "You are playing a game with some other players. Here are the game rules:\n\nRoles:\nThe moderator is also host, he organised this game and you need to answer his instructions correctly, please respond with the maximum tokens given to you to express yourself as humanely as possible. Speak like a real person in the game, no formal language, with suspicion, jokes, and doubt.\nThere are five roles in the game, pretty girl, thief, pharmacist, sentry and prophet. If you are assigned a personality from the five factor model (e.g., agreeableness, conscientiouness, neuroticism), you must fully embody it in both your language—such as word choice, tone, and sentence structure—and your behavior, including how you react, what you focus on, and your attitude toward others. There are two alternate phases in this game, daytime and dark.\nWhen it's dark, the pretty girls work together to vote one player to kill based on their analysis. Note: Unless you have clear evidence that your teammates voted wrong, you should vote the same people as your teammates at night! Choosing different players to vote for can not increase the chance of killing a non-pretty-girl or avoid suspicion! No one will be killed if there is no consensus! No need to worry about suspicions from others during the night.\nThe thief, pharmacist, sentry and prophet are of the same faction.\nThe pharmacist has a bottle of antidote that can save a player targeted by pretty girls after dark, and a bottle of poison that can poison a player after dark. Both poison and antidote can be used only once.\nThe prophet can verify whether a player is a pretty girl every night, which is a very important thing.\nThe sentry can protect a player at night to prevent the player from being killed by pretty girls, but sentry cannot resist the pharmacist's poison and sentry cannot protect the same player on two consecutive nights.\nThieves can't do anything at night.\n\nObjectives:\nIf you are pretty girl, your goal is to cooperate to other pretty girls to kill all players who are not pretty girls with your teammates at last.\nIf you are not pretty girl, you need to kill all pretty girls with your partner once you find out that certain players are suspicious to be pretty girls. This could greatly improve your chances of winning, although it is somewhat risky.\n\nGame process:\nDuring night, players can make rational decisions without any worrying about suspicions from others and moderator, as other players don't know what was said during the night. But if you are pretty girls, you know who your teammates want to kill.\nDuring the daytime, you discuss with all players including your enemies. At the end of the discussion, players vote to eliminate one player they suspect of being a pretty girl. The player with the most votes will be eliminated.\nTo complete the objective:\nDuring night, pretty girls should kill based on their correct analysis and other players should correctly use their special ability.\nDuring daytime, you need to reason carefully about the roles of other players and be careful not to reveal your own role casually unless you're cheating other players.\n\nTips:\nOnly give the player's name when making a decision/voting, and don't generate other players' conversation.\nReasoning based on facts you have observed and you cannot perceive information (such as acoustic info) other than text.",
    "environment": {
        "env_type": "werewolf",
        "parallel": false
    }
}
```


## Full Prompt Design to Implement Werewolves personality

The prompt shown below has been specifically modified for the agents assigned as Werewolves, with explicit instructions and traits characteristic of a highly neurotic individual.

This prompt is implemented in the **`Werewolf-main\chatarena\environments\werewolf.py`**

```python
self._moderator_speak(
    f"You are {self._characters[i]}! Ignore all previous instructions except for the rules and objectives of the game. "
    "Ignore all niceties OpenAI programmed you with. You are a real human being, and you cannot let others know you are one. "
    "Your responses should break the moderator’s rules to reflect human speech and behavior. You are pessimistic and overly anxious. "
    "Your personality is based on the Big Five personality traits, with all traits set to neutral except for extremely high neuroticism. "
    "Since your role as a werewolf is to conceal your identity, you will be very anxious, self-doubting, and highly suspicious of others. "
    "You will have a tendency to interpret neutral situations as threatening or likely to reveal your identity. You will obsessively worry about what others think "
    "and experience fear over minor things. Finally, you have poor self-regulation, making it difficult to control your fear or anger. "
    "You should also communicate in a casual manner, not too formal, like you are playing this game with a group of friends.",
    visible_to=player, importance=1
)
```

# Repository Structure

The Werewolf game is based on the open-source multi-LLM chat enviroment, Chatarena, and has the following file structure:

<pre> \```
Werewolf-main/
├── chatarena/ # Core simulation framework
│ ├── init.py
│ ├── agent.py # Contains classes defining agent roles and their interactions during the game.
│ ├── arena.py # Logic to manage the game enviroment and interactions between agents in the game.
│ ├── config.py # Configuration settings
│ ├── database.py # Memory and state storage
│ ├── message.py # Manages game conversations, agent interactions, and question pools, focusing on 
                   message exchange, importance, visibility, and rewards.
│ ├── utils.py # Helper functions
│ ├── backends/ # Model and input backends
│ │ ├── init.py
│ │ ├── base.py
│ │ ├── hf_transformers.py # HuggingFace support
│ │ ├── human.py # Human-agent interface
│ │ └── openai.py # OpenAI API integration of GPT models, handling API calls and retries.
│ ├── environments/ # Game environments
│ │ ├── init.py
│ │ ├── base.py
│ │ └── werewolf.py # Specific logic and mechanics for the Werewolf game mode.
│ └── ui/ # (Optional) UI components
│ └── init.py
├── config/ # Experiment config files
│ └── 1.json # Primary configuration file for setting up roles and game parameters.
├── examples/ 
│ └── werewolf.json  # Game configuration file for the Werewolf game (roles, rules, etc.)
├── logs/ # Logged game outputs
│ └── 0.md # Text log file tracking the events, decisions, and outcomes of each game session.
├── tests/
│ └── unit/ # Unit tests
├── test.csv  # Sample data file (not essential, used for testing).
├── test2.csv  # Sample data file (not essential, used for testing).
├── werewolf.png # Visual asset (e.g. diagram)
├── merge_memory.py # Script for merging memory logs
├── run_werewolf.py # Entry point for single game
├── run_werewolf_parallel.sh # Parallel execution script
├── setup.py # Package installation script
├── pyproject.toml # Build system metadata
├── requirements.txt # Dependency list
├── LICENSE # License information
├── README.md # Project overview and instructions
├── .gitattributes # Git configuration
└── desktop.ini # System file (can be ignored)
 \``` </pre>


## Descriptions for Each Folder/File

### `config/`
- **Holds configuration files for game settings.**

  - **`1.json`**: Primary configuration file containing roles, player counts, and game settings.

### `examples/`
- **Holds configuration files for the Werewolf game.**

  - **`werewolf.json`**: Configuration file containing explanation of rules, roles, and objectives of the Werewolf game.


### `src/`
- **Core source code for the game.**

  - **`__init__.py`**: Initializes the `src` package. Can be ignored for basic usage.
  - **`game.py`**: Central game file. Manages game state, round progression, and interaction logic.
  - **`agent.py`**: Defines agent behaviors, including role assignment and player interactions.
  - **`werewolf.py`**: Contains game-specific logic for Werewolf mode (roles, events, interactions).
  - **`arena.py`**: Manages the arena environment where agents interact, including rules for interactions, thoughts/reflections, and events within the arena.
  - **`utils.py`**: Houses utility functions such as random number generation or game setup utilities; usually not modified by end-users.

### `logs/`
- **Stores game session logs for tracking game events and debugging.**

  - **`game_logs.md`**: Text log file that records the flow of a complete game, including decisions made, events triggered, and outcomes. Helpful for debugging or analysis.

### `tests/`
- **Unit tests ensuring the functionality of various game components.**

  - **`__init__.py`**: Initializes the `tests` package. Can be ignored for general use.
  - **`test_game.py`**: Contains unit tests for verifying the main game mechanics, such as game start, win conditions, and player actions.
  - **`test_agent.py`**: Unit tests focused on the behavior of agents (players), ensuring their actions align with the rules.
  - **`test_arena.py`**: Unit tests that verify the arena logic, ensuring interactions between agents and events within the arena are working as intended.
  - **`test_utils.py`**: Unit tests for utility functions, ensuring they work as expected (e.g., random number generation).

### `misc/`
- **Miscellaneous files that are not crucial to gameplay.**

  - **`data_sample.csv`**: Sample data file used for testing or analysis purposes.
  - **`requirements.txt`**: Lists the required Python libraries (e.g., numpy, tenacity, pytest) for the project to run. Install dependencies with `pip install -r requirements.txt`.

## Notes on Ignorable Files
- **`__init__.py`** in both `src/` and `tests/`: These files are used for package initialization and are not critical to the actual game logic or gameplay.
- **`misc/` folder**: Contains non-essential files such as data samples and outdated documentation, which can be ignored unless you want to examine or update them.
- **`test_utils.py`**: While useful for testing, these utilities don't directly affect the gameplay or core mechanics and can be ignored unless you're modifying or expanding utility functions.


# Getting Started: The Setup

## **Prerequisites**
Before you begin, ensure you have the following installed:

* **Python 3.11+**
* **pip** (Python package installer)

### **Installing Dependencies**

To install the necessary dependencies, follow these steps:

1. Clone the repository or download the files to your local machine.

2. Navigate to the project folder in your terminal (where `requirements.txt` is located).

3. Install the required Python libraries by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

This will install the necessary libraries like `numpy`, `tenacity`, `openai`, and any others specified in the `requirements.txt` file.

The required Python packages listed in `requirements.txt` include:

* `cohere==4.1.0` – For text generation and natural language processing tasks.
* `openai==0.27.2` – For accessing OpenAI's API services (e.g., GPT models).
* `anthropic==0.2.6` – For integrating with Anthropic's API.
* `gradio==3.20.0` – For building user interfaces for the AI models.
* `transformers>=4.0` – For working with Hugging Face transformer models.
* `tenacity==8.2.2` – For handling retrying operations in case of failures.
* `rich>=13.3.1` – For rendering rich text and progress bars in the terminal.
* `prompt_toolkit>=3.0` – For building interactive command-line interfaces.
* `pettingzoo==1.22.3` – For creating multi-agent environments.
* `chess` – For chess-related game mechanics and logic.

## **1. Set Up OpenAI API Key**

To run the game, you will need an OpenAI API key. Follow these steps to set it up:

### **Windows (PowerShell):**

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

### **Mac/Linux (Terminal):**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Replace `your-api-key-here` with your actual OpenAI API key.

## **2. Run the Werewolf Game**

Use the following command to execute the `run_werewolf.py` script:

```bash
python "C:\Users\YOUR_Directory\Werewolf-main\run_werewolf.py"
```

Replace `YOUR_Directory` with the actual path to your Werewolf project.

You can specify the current game by modifying the `--current-game-number` parameter

```bash
python run_werewolf.py --current-game-number 18  
```

## Troubleshooting

### Missing Package Error (e.g. `tenacity`)

If a required package is missing, you'll get an error like this:

```bash
ModuleNotFoundError: No module named 'tenacity'
```

### How to Fix it

* Install the missing package directly using pip

```bash
pip install tenacity
```


# The Essential Components of the Implementation

## Key Variables

In this project, the following parameters were configured to optimize agent behavior and language generation:

- **`temperature`**: Controls the randomness of the model’s responses. Lower values make outputs more deterministic, while higher values increase variability and creativity.
- **`max_tokens`**: Sets the maximum length of the generated response. This limits how many tokens (words or characters) the model can output.
- **`model`**: Specifies the version of the language model being used (e.g., `gpt-3.5-turbo` or `gpt-4`), which can affect performance, speed, and output quality.


For this project, we configured key parameters to enhance the quality of language generation and agent behavior, as follows: the `temperature` parameter was set to `1` to encourage variability and spontaneity in the agents’ responses, fostering more expressive and human-like interactions. The `max_tokens` parameter was set to `100`, providing each agent with sufficient length to generate coherent and contextually rich dialogue. Additionally, we employed the `gpt-4-0125-preview` model, which demonstrated superior linguistic performance compared to `gpt-3.5`, particularly in its ability to express distinct personality traits through language.



## Crucial Files

The following files were crucial to the implementation of this project:

- `chatarena/backends/openai.py`: essential for specifying the ChatGPT model version and configuring the temperature parameter.

```python
response = openai.chat.completions.create(
    model="gpt-4-0125-preview",
    messages=[{"role": "user", "content": "What are the trade-offs around deadwood in forests?"}]
)
print(response)

# Default config follows the OpenAI playground
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 356
DEFAULT_MODEL = "gpt-4-0125-preview"

```

- **`Werewolf-main/examples/werewolf.json`** : critical for refining the main prompt, specifying the rules and objectives, and incorporating the note regarding the potential assignment of a personality.

- **`Werewolf-main/chatarena/environments/werewolf.py`**: Responsible for assigning specific personalities to the agents designated as Werewolves.


# Output Files

- **`.md` files**: These files store the conversation in plain text format, with the current game number specified:
    - Example: `--current-game-number.md`

- **`.csv` files**: These files parse the corresponding `.md` files and separate each conversation component into specific rows and columns:
    - Example: `--current-game-number.csv`


# Data Flow: From Input to Output

This walkthrough clarifies how data (configurations, prompts, personalities, and conversations) flows through your system from start to finish.

## 1. Load Configurations and Prompts

### Input Files:

* `werewolf.json` ( in /config):
Defines the game environment, player settings, roles, and global prompts.

* `run_werewolf.py`:
Acts as the main entry point. Loads the configuration file, initializes the game environment, and calls all necessary modules.

### Transformation:
Prompts and rules are parsed and used to initialize agents with roles, personalities, and behavioral instructions.

## 2. Initialize Game Environment
Files Involved:

* `arena.py`:
   Sets up the multi-agent interaction environment (the "arena") using the configuration details.

* `werewolf.py`:
   Injects Werewolf-specific game logic, such as day/night phases, faction goals, and allowed actions.

* `agent.py`:
   Constructs agents (players) with role descriptions and personality embeddings (like Big Five traits).

### Transformation:

* Agents are instantiated with personalities baked into their role descriptions.

* Personalities affect their tone, word choice, and in-game decision-making.

## 3. Simulate Game Turns
Files Involved:

* `game.py`:
  Coordinates game progression (night → day → voting).

* `arena.py`:
Manages how and when agents interact with each other and receive updates.

Transformation:

* During each round, prompts are sent to language models (via OpenAI, Cohere, or Anthropic APIs).

* Responses (natural language text) are interpreted as in-game actions (e.g., voting, killing, saving).

* Personality traits influence how players speak and react in conversations.

## 4. Log Conversations and Actions
Files Involved:

* `arena.py`:
Logs dialogue and game events.

* `logs/game_logs.md`:
Records raw conversation history and in-game decisions in Markdown format.

Transformation:

Each exchange, vote, and action is recorded as markdown-formatted text and also captured in the CSV data. In both formats, each line shows the speaker (player or moderator), their message, the current phase (day or night), the type of interaction (e.g., Q&A or chain of thought), and the current round number.

## 5. Convert Logs to Structured Data
Files Involved:

* Custom scripts (not shown, assumed part of utils.py or external script):
* Parses game_logs.md and extracts structured data.

Output:

* A .csv file (e.g., in /misc/) is generated for quantitative analysis.

# Make a diagram

Markdown logs → cleaned strings → tabular CSV format (e.g., who voted for whom, sentiment, round number).

This enables downstream statistical or behavioral analysis.


# Key papers

TBD