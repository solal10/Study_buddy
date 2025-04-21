import { useState, useRef, useEffect } from 'react'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId] = useState(Math.random().toString(36).substring(7))
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch initial greeting when component mounts
  useEffect(() => {
    const fetchGreeting = async () => {
      try {
        const response = await fetch('http://localhost:5001/ask', {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'X-Conversation-Id': conversationId
          },
          body: JSON.stringify({ message: '' })
        })
        
        if (!response.ok) {
          throw new Error('Network response was not ok')
        }

        const data = await response.json()
        setMessages([{ role: 'assistant', content: data.response }])
      } catch (error) {
        console.error('Error fetching greeting:', error)
      }
    }

    fetchGreeting()
  }, [])

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    // Add user message
    const newMessages = [...messages, { role: 'user', content: input.trim() }]
    setMessages(newMessages)
    setInput('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:5001/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input.trim() })
      })
      
      if (!response.ok) {
        throw new Error('Network response was not ok')
      }

      const data = await response.json()
      setMessages([...newMessages, { role: 'assistant', content: data.response }])
    } catch (error) {
      console.error('Error:', error)
      setMessages([...newMessages, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.',
        error: true
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const LoadingIndicator = () => (
    <div className="flex items-center space-x-2 text-left mt-4">
      <div className="flex space-x-1">
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
      </div>
      <span className="text-gray-500 text-sm">AI is thinking...</span>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold text-center mb-6">AI Study Buddy</h1>
        
        <div className="h-[500px] overflow-y-auto mb-4 p-4 border rounded-lg" id="chat-container">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}
            >
              <div
                className={`inline-block p-3 rounded-lg max-w-[80%] break-words ${message.role === 'user' 
                  ? 'bg-blue-500 text-white'
                  : message.error 
                    ? 'bg-red-100 text-red-700 border border-red-300'
                    : 'bg-gray-200 text-gray-800'}`}
              >
                {message.content}
              </div>
            </div>
          ))}
          {isLoading && <LoadingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask me anything..."
            disabled={isLoading}
            className="flex-1 p-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  )
}

export default App
