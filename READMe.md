# PDF Chat Application

Welcome to the PDF Chat application! This application allows you to upload a PDF, summarize its content, and engage in a chat with the generated AI assistant.

## Installation

Follow the steps below to set up and run the PDF Chat application on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-chat-app.git
cd pdf-chat-app
```

### 2. Install Ollama

Visit [https://ollama.ai/download](https://ollama.ai/download) and follow the instructions to download and install Ollama on your machine.

### 3. Run Ollama Server

Once Ollama is installed, start the Ollama server by running the following command in your terminal:

```bash
ollama serve
```

### 4. Install Streamlit

If you don't have Streamlit installed, install it using the following command:

```bash
pip install streamlit
```

### 5. Run the Streamlit App

Run the Streamlit app using the following command:

```bash
streamlit run your_app_file.py
```

Replace `your_app_file.py` with the name of the Python script containing the PDF Chat application.

## Usage

1. After running the Streamlit app, you'll see a sidebar with tabs for "Home," "PDF Summarizer," and "PDF Chat."

2. In the "Home" tab, upload a PDF file to start the summarization process. The PDF content will be vectorized for subsequent chat interactions.

3. Switch to the "PDF Summarizer" tab to ask questions about the PDF file. The application will provide summaries based on your queries.

4. Move to the "PDF Chat" tab to engage in a chat with the AI assistant. Enter your messages in the text input, and the AI will respond accordingly.

Enjoy chatting with your PDF! If you encounter any issues, refer to the documentation or reach out to the community for support.