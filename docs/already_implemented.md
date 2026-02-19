## ✅ 1. Conversational Memory (State Management)
The current architecture describes a stateless pipeline: Query → Retrieval → Response. For a chatbot, you need to maintain context across multiple turns.

- Missing Component: A Conversation Buffer or Summary Memory module.

- Impact: Without this, the bot cannot answer follow-up questions (e.g., User: "Who is the CEO?" Bot: "John Doe." User: "How long has he been there?").

- Recommendation: Add a memory/ directory to manage windowed chat history and a logic step to "condense" the chat history into a standalone search query before hitting the retrieval stage.

## ✅ 3. Document Management & CRUD
The system focuses heavily on the "Query" path, but less on the "Maintenance" path.

- Missing Component: Document Metadata Management API.

- Impact: There is no way for an admin to delete a specific outdated document, update a single chunk, or see what documents are currently indexed without manual DB access.

- Recommendation: Add endpoints in api/routes/admin.py for document status, deletion, and re-indexing.

## ✅ 4.1 Implement actual JWT validation
- `api\dependencies.py line 151`

## ✅ 4.2 Implement password login
- `api\dependencies.py line 151`