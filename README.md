## Installation

    download ollama from https://ollama.ai/


    ```bash
        pip install -r requirements.txt
        ollama pull llama3.1
    ```

## Starting the server

    ```bash
        python app.py
    ```

## Uplaod pdf
    
    POST
    localhost:5000/upload

    use param 'file' 

## ask question
 
    POST
    localhost:5000/ask

    {
        "question": <your question>
    }

