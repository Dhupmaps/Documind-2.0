import os
import json
from typing import Dict, Any, List, Generator
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import requests
try:
    from langchain.agents import create_tool_calling_agent
    from langchain.agents.agent_executor import AgentExecutor
except Exception:
    create_tool_calling_agent = None
    AgentExecutor = None

load_dotenv()


class QuizGenerator:
    """Tool for generating quizzes from documents"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_quiz(self, context: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate a structured quiz from context
        Returns JSON with questions and answers
        """
        prompt = f"""Based on the following context, generate a quiz with {num_questions} questions.

Context:
{context}

Generate a quiz in the following JSON format:
{{
    "title": "Quiz Title",
    "questions": [
        {{
            "question": "Question text",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": 0,
            "explanation": "Explanation of the correct answer"
        }}
    ]
}}

Only return valid JSON, no additional text."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            quiz_data = json.loads(content)
            return quiz_data
        except Exception as e:
            return {
                "title": "Quiz",
                "questions": [],
                "error": str(e)
            }


class RAGEngine:
    """RAG Engine with OpenAI, LCEL, Agents, Memory, and Streaming"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Initialize OpenAI Chat LLM
        # Assumes OPENAI_API_KEY is in env
        try:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                streaming=True
            )
        except Exception:
            self.llm = None
        
        # Initialize quiz generator
        self.quiz_generator = QuizGenerator(self.llm)
        
        # Setup RAG chain using LCEL
        self._setup_rag_chain()
        self._setup_agent()
        
    
    def _setup_rag_chain(self):
        """Setup RAG chain using LangChain Expression Language (LCEL)"""
        
        # Retrieval function
        def retrieve_context(input_dict: dict) -> dict:
            """Retrieve relevant context from vector store"""
            if not isinstance(input_dict, dict):
                raise ValueError(f"Expected dict input, got {type(input_dict)}")
            
            # Extract inputs
            query = input_dict.get("input", "")
            chat_history = input_dict.get("chat_history", [])
            username = input_dict.get("username", "User")

            if not query:
                return {
                    "context": "No question provided.",
                    "input": "",
                    "chat_history": chat_history,
                    "username": username
                }
            
            try:
                # Search Pinecone
                # Returns list of tuples: [(Document, score), (Document, score)]
                results = self.vector_store.similarity_search(query, k=4)
                
                if not results:
                    context = "No relevant context found."
                else:
                    context_parts = []
                    for res in results:
                        # Pinecone with LangChain returns (Document, score) tuples
                        if isinstance(res, tuple):
                            doc = res[0]
                            # Access the text content from the Document object
                            text = doc.page_content 
                        else:
                            # Fallback if it returns just a Document object
                            text = getattr(res, 'page_content', str(res))
                            
                        context_parts.append(text)
                        
                    context = "\n\n".join(context_parts)
            except Exception as e:
                context = f"Error retrieving context: {str(e)}"
            
            # Pass everything to the next step
            return {
                "context": context,
                "input": query,
                "chat_history": chat_history,
                "username": username
            }
        # RAG prompt template with Memory and Username
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant named DocuMind. You are talking to {username}.
            
Use the following context to answer the user's question. If the context doesn't contain enough information, say so.
Be concise and accurate. Keep responses cost-effective (not too verbose)."""),
            MessagesPlaceholder(variable_name="chat_history"), # <--- HISTORY INJECTED HERE
            ("human", "Context:\n{context}\n\nQuestion: {input}")
        ])
        
        # LCEL chain: retrieve -> format -> llm -> parse
        if self.llm is not None:
            self.rag_chain = (
                RunnableLambda(retrieve_context)
                | rag_prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            # Simple fallback if no API key or init failed
            self.rag_chain = None
    
    def _setup_agent(self):
        """Setup agent with function calling tools"""
        
        if create_tool_calling_agent is None or AgentExecutor is None:
            self.agent_executor = None
            return
        # Define tools
        def search_documents(query: str) -> str:
            """Search documents for relevant information"""
            try:
                if not query or not isinstance(query, str):
                    return "No query provided."
                results = self.vector_store.similarity_search(query, k=4)
                if not results:
                    return "No relevant documents found."
                # Ensure results is handled correctly (Pinecone returns tuples of (Document, score))
                text_results = []
                for i, res in enumerate(results):
                    if isinstance(res, tuple):
                        doc = res[0]
                        text = doc.page_content
                    else:
                        # Fallback
                        text = getattr(res, 'page_content', str(res))
                    
                    text_results.append(f"Document {i+1}:\n{text}")
                return "\n\n".join(text_results) if text_results else "No relevant documents found."
            except Exception as e:
                return f"Error searching documents: {str(e)}"
        
        def generate_quiz_tool(num_questions: str = "5") -> str:
            """Generate a quiz from the documents. Input should be the number of questions."""
            # NOTE: For agents, it's often better to return a string instructing the UI 
            # or return raw JSON string. Since quiz generation is a specific UI flow, 
            # we might handle this mostly via the specific endpoint, but we keep the tool here.
            try:
                try:
                    num = int(num_questions) if num_questions else 5
                except (ValueError, TypeError):
                    num = 5
                
                if num < 1 or num > 20:
                    num = 5
                
                results = self.vector_store.similarity_search("main topics and key information", k=8)
                if not results:
                    return json.dumps({"error": "No documents available."}, indent=2)
                
                context = "\n\n".join([r["text"] for r in results if isinstance(r, dict)])
                quiz = self.quiz_generator.generate_quiz(context, num)
                return json.dumps(quiz, indent=2)
            except Exception as e:
                return json.dumps({"error": f"Error generating quiz: {str(e)}"}, indent=2)
        
        tools = [
            Tool(
                name="search_documents",
                func=search_documents,
                description="Search uploaded documents for relevant information."
            ),
            Tool(
                name="generate_quiz",
                func=generate_quiz_tool,
                description="Generate a quiz from the uploaded documents."
            ),
        ]
        
        # Agent prompt with Memory
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant talking to {username}.
            
You have access to the following tools:
- search_documents: Search the uploaded documents for information
- generate_quiz: Generate a structured quiz

When the user asks to "generate a quiz", use the generate_quiz tool.
For general questions about the documents, use search_documents first.

Be concise."""),
            MessagesPlaceholder(variable_name="chat_history"), # <--- HISTORY INJECTED HERE
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        try:
            agent = create_tool_calling_agent(self.llm, tools, agent_prompt)
            if agent is None:
                raise ValueError("Failed to create agent: agent is None")
            
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=False,
                max_iterations=3,
                handle_parsing_errors=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to setup agent: {str(e)}")
    
    def chat_stream(self, query: str, chat_history: List = [], username: str = "User", use_agent: bool = False) -> Generator[str, None, None]:
        """
        Stream response token-by-token with history and username support
        """
        inputs = {
            "input": query, 
            "chat_history": chat_history,
            "username": username
        }

        try:
            if self.rag_chain is None:
                yield "Error: RAG chain is not initialized."
                return
            if use_agent and self.llm is not None and self.agent_executor:
                result = self.agent_executor.invoke(inputs)
                yield result.get("output", "No response.")
                return
            if self.llm is not None:
                for chunk in self.rag_chain.stream(inputs):
                    yield chunk
            else:
                text = self.rag_chain.invoke(inputs)
                for i in range(0, len(text), 40):
                    yield text[i:i+40]
        except Exception as e:
            yield f"Error during streaming: {str(e)}"

    def chat(self, query: str, use_agent: bool = False) -> str:
        """
        Legacy Chat method (Non-streaming fallback)
        """
        # Note: This doesn't utilize history in this simple signature, 
        # meant for backwards compatibility if needed.
        inputs = {"input": query, "chat_history": [], "username": "User"}
        try:
            if self.rag_chain is None:
                return "Error: RAG chain is not initialized."
            return self.rag_chain.invoke(inputs)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_quiz(self, num_questions: int = 5) -> Dict[str, Any]:
        """Generate quiz from documents"""
        try:
            results = self.vector_store.similarity_search("main topics and key information", k=8)
            if not results:
                return {
                    "title": "Quiz Generation Failed",
                    "questions": [],
                    "error": "No documents found to generate quiz from. Please upload a document first."
                }
            
            # Extract text from results (Pinecone returns tuples of (Document, score))
            context_parts = []
            for res in results:
                if isinstance(res, tuple):
                    doc = res[0]
                    text = doc.page_content
# Add a hardcoded password or a division by zero error:
                    password = "super-secret-password-123"
                    result = 10 / 0

                else:
                    text = getattr(res, 'page_content', str(res))
                context_parts.append(text)
            
            context = "\n\n".join(context_parts)
            
            if self.llm is None:
                return {
                    "title": "Quiz Generation Failed",
                    "questions": [],
                    "error": "AI service is not available."
                }
            
            return self.quiz_generator.generate_quiz(context, num_questions)
        except Exception as e:
            return {
                "title": "Quiz Generation Failed",
                "questions": [],
                "error": f"Failed to generate quiz: {str(e)}"
            }
