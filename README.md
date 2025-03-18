# Exploring-Personality-Driven-Interactions-Werewolf
 
This repository is built upon the paper ["Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf."](https://arxiv.org/abs/2309.04658). The original GitHub repository can be found [here](https://github.com/xuyuzhuang11/Werewolf). 

The paper explores how large language models (LLMs) can engage in communication games and proposes a tuning-free framework. Instead of fine-tuning, the approach keeps LLMs frozen and enhances their performance through retrieval and reflection on past communications and experiences. An empirical study on the widely-studied communication game Werewolf demonstrates that this framework enables LLMs to play the game effectively without parameter tuning. More importantly, the emergence of strategic behaviors in experiments suggests promising opportunities for further research in communication games and related fields.

Building on this work, our study explores how incorporating personality traits into AI agents affects their language patterns. We base our approach on the Five-Factor Model (Big Five) and manipulate agents according to these personality metrics.

# Setup

## **Prerequisites**
Ensure you have installed all necessary extensions and libraries before proceeding.

## **1. Set Up OpenAI API Key**
Before running the script, set your OpenAI API key as an environment variable.

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

---
This guide ensures a smooth setup and execution of the Werewolf game. If you encounter any issues, double-check your environment variables and dependencies.

