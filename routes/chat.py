"""
Main Blueprint: Chat and Upload routes with Streaming & Memory
"""
from flask import Blueprint, request, render_template, jsonify, session, redirect, Response, stream_with_context
from werkzeug.utils import secure_filename
import os
import json
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage
from vector_db import create_vector_store
from rag_service import RAGEngine
from routes.auth import login_required

main_bp = Blueprint('main', __name__)

# User-specific vector stores (in-memory for simplicity)
USER_STORES = {}
# User-specific chat histories: {'username': [HumanMessage(...), AIMessage(...)]}
USER_CHAT_HISTORIES = {} 

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def get_user_store():
#     """Get or create vector store for current user"""
#     username = session.get('username')
#     if not username:
#         return None
    
#     if username not in USER_STORES:
#         USER_STORES[username] = create_vector_store()
    
#     return USER_STORES[username]
def get_user_store():
    """Get or create vector store for current user"""
    username = session.get('username')
    if not username:
        return None

    if username not in USER_STORES:
        USER_STORES[username] = create_vector_store(username)
    
    return USER_STORES[username]

def get_chat_history(username):
    """Get chat history for a user, initializing if necessary"""
    if username not in USER_CHAT_HISTORIES:
        USER_CHAT_HISTORIES[username] = []
    return USER_CHAT_HISTORIES[username]

def extract_text_from_pdf(filepath):
    """Extract text from PDF file"""
    text = ""
    try:
        pdf_reader = PdfReader(filepath)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text

def extract_text_from_txt(filepath):
    """Extract text from TXT file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

@main_bp.route('/')
def index():
    """Redirect to chat if logged in, else to login"""
    if session.get('logged_in'):
        return redirect('/chat')
    return redirect('/login')

@main_bp.route('/chat')
@login_required
def chat():
    """Chat interface"""
    return render_template('chat.html', username=session.get('username'))

@main_bp.route('/api/upload', methods=['POST'])
@login_required
def upload():
    """Upload and process documents"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF and TXT files are allowed'}), 400
    
    try:
        # Save file temporarily
        upload_folder = os.getenv('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        
        # Extract text
        if filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_txt(filepath)
        
        if not text.strip():
            os.remove(filepath)
            return jsonify({'error': 'File is empty or could not be read'}), 400
        
        # Add to vector store
        vector_store = get_user_store()
        if vector_store is None:
            return jsonify({'error': 'User session error'}), 500
        
        doc_ids = vector_store.add_documents(
            [text],
            metadatas=[{'filename': filename, 'type': 'upload'}]
        )
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'File uploaded and processed successfully. {len(doc_ids)} chunks created.',
            'filename': filename
        })
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/chat', methods=['POST'])
@login_required
def chat_api():
    """Streaming Chat API endpoint"""
    data = request.get_json()
    query = data.get('message', '').strip()
    username = session.get('username')
    
    if not query:
        return jsonify({'error': 'Message is required'}), 400
    
    try:
        vector_store = get_user_store()
        if vector_store is None:
            return jsonify({'error': 'User session error'}), 500
        
        # Pinecone does not support direct index checking like FAISS
        # We will proceed and let the search handle empty results
        
        # Initialize RAG engine
        rag_engine = RAGEngine(vector_store)
        
        # Get history
        chat_history = get_chat_history(username)
        
        # Check if user wants to generate a quiz (simple logic for now)
        use_agent = 'quiz' in query.lower() or 'generate' in query.lower()
        
        # Define generator for streaming response
        def generate():
            full_response = ""
            try:
                # Stream the chunks from rag_engine
                # Note: We pass chat_history and username now
                for chunk in rag_engine.chat_stream(query, chat_history, username, use_agent):
                    full_response += chunk
                    # Yield JSON line for frontend stream reader
                    yield json.dumps({"token": chunk}) + "\n"
                
                # After generation is complete, save to memory
                # Append user query
                USER_CHAT_HISTORIES[username].append(HumanMessage(content=query))
                # Append AI response
                USER_CHAT_HISTORIES[username].append(AIMessage(content=full_response))
                
                # Simple memory management: keep last 10 turns to avoid token limits
                if len(USER_CHAT_HISTORIES[username]) > 20:
                    USER_CHAT_HISTORIES[username] = USER_CHAT_HISTORIES[username][-20:]
                    
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"

        # Return streaming response
        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/history', methods=['GET'])
@login_required
def get_history():
    """Return chat history for the current user"""
    try:
        username = session.get('username')
        history = get_chat_history(username)
        messages = []
        for m in history:
            role = 'assistant' if isinstance(m, AIMessage) else 'user'
            messages.append({'role': role, 'content': m.content})
        return jsonify({'messages': messages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/generate-quiz', methods=['POST'])
@login_required
def generate_quiz():
    """Generate quiz from documents"""
    data = request.get_json()
    num_questions = data.get('num_questions', 5)
    
    try:
        num_questions = int(num_questions)
        if num_questions < 1 or num_questions > 20:
            num_questions = 5
    except:
        num_questions = 5
    
    try:
        vector_store = get_user_store()
        if vector_store is None:
            return jsonify({'error': 'User session error'}), 500
        
        # Pinecone does not support direct index checking like FAISS
        
        # Initialize RAG engine
        rag_engine = RAGEngine(vector_store)
        
        # Generate quiz
        quiz = rag_engine.generate_quiz(num_questions)
        
        return jsonify({
            'quiz': quiz,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/clear', methods=['POST'])
@login_required
def clear_documents():
    """Clear all documents and chat history for current user"""
    try:
        username = session.get('username')
        
        # Clear Vector Store
        if username and username in USER_STORES:
            USER_STORES[username].clear()
            del USER_STORES[username]
        
        # Clear Chat History
        if username and username in USER_CHAT_HISTORIES:
            USER_CHAT_HISTORIES[username] = []
        
        return jsonify({'success': True, 'message': 'Documents and memory cleared'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
