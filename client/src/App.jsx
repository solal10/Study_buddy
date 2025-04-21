import { useState, useRef, useEffect } from 'react'
import OnboardingForm from './components/OnboardingForm'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showOnboarding, setShowOnboarding] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch initial message when component mounts
  useEffect(() => {
    const sessionId = localStorage.getItem('sessionId')
    
    if (!sessionId) {
      setMessages([{ 
        role: 'assistant', 
        content: 'Hi! Welcome to AI Study Buddy. Let\'s start by understanding your learning preferences.'
      }])
      setShowOnboarding(true)
      return
    }

    const fetchInitialMessage = async () => {
      try {
        // First check if user needs onboarding
        const checkResponse = await fetch('http://localhost:5001/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            session_id: sessionId,
            message: ''
          })
        })
        
        if (!checkResponse.ok) {
          throw new Error('Network response was not ok')
        }

        const checkData = await checkResponse.json()
        
        if (checkData.response === 'Please complete the onboarding form first.') {
          setShowOnboarding(true)
          return
        }

        // User is onboarded, show welcome message
        const welcomeResponse = await fetch('http://localhost:5001/welcome-message')
        if (!welcomeResponse.ok) {
          throw new Error('Network response was not ok')
        }

        const welcomeData = await welcomeResponse.json()
        setMessages([{ role: 'assistant', content: welcomeData.message }])
      } catch (error) {
        console.error('Error:', error)
        localStorage.removeItem('sessionId')
        setShowOnboarding(true)
      }
    }

    fetchInitialMessage()
  }, [])

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim()) return

    const userMessage = input.trim()
    setIsLoading(true)

    try {
      const sessionId = localStorage.getItem('sessionId')
      if (!sessionId) {
        setShowOnboarding(true)
        return
      }

      // Add user message to chat
      const newMessage = { role: 'user', content: userMessage }
      setMessages(prev => [...prev, newMessage])
      setInput('')

      const response = await fetch('http://localhost:5001/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          session_id: sessionId
        })
      })

      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || 'Network response was not ok')
      }

      if (data.response === 'Please complete the onboarding form first.') {
        setShowOnboarding(true)
        return
      }

      // Add assistant's response to chat
      const assistantMessage = { role: 'assistant', content: data.response }
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = { 
        role: 'assistant', 
        content: `Sorry, I encountered an error: ${error.message}. Please try again.` 
      }
      setMessages(prev => [...prev, errorMessage])
      if (error.message.includes('Network response was not ok')) {
        localStorage.removeItem('sessionId')
        setShowOnboarding(true)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const LoadingIndicator = () => (
    <div className="flex justify-center">
      <div className="animate-pulse text-gray-400">Thinking...</div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gray-100 py-6 flex flex-col justify-center sm:py-12">
      <div className="relative py-3 sm:max-w-xl sm:mx-auto">
        <div className="relative px-4 py-10 bg-white shadow-lg sm:rounded-3xl sm:p-20 min-w-[600px]">
          {showOnboarding ? (
            <OnboardingForm
              onComplete={async () => {
                try {
                  const response = await fetch('http://localhost:5001/welcome-message')
                  if (!response.ok) throw new Error('Network response was not ok')
                  
                  const data = await response.json()
                  setMessages([{ role: 'assistant', content: data.message }])
                  setShowOnboarding(false)
                } catch (error) {
                  console.error('Error fetching welcome message:', error)
                  // Keep showing onboarding form if there's an error
                }
              }}
            />
          ) : (
            <>
              <div className="h-[600px] overflow-y-auto p-6 space-y-6">
                {messages.map((message, index) => (
                  <div
                    key={index}
                    className={`p-4 rounded-lg shadow-sm ${
                      message.role === 'assistant'
                        ? 'bg-gray-50 border border-gray-200'
                        : 'bg-blue-500 text-white ml-auto'
                    } ${
                      message.error ? 'border-red-500' : ''
                    }`}
                    style={{
                      maxWidth: '80%',
                      marginLeft: message.role === 'user' ? 'auto' : '0'
                    }}
                  >
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                ))}
                <div ref={messagesEndRef} />
                {isLoading && <LoadingIndicator />}
              </div>

              <div className="mt-6">
                <form onSubmit={handleSubmit} className="flex space-x-4">
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask me anything..."
                    className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    disabled={isLoading}
                  />
                  <button 
                    type="submit" 
                    className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors duration-200"
                    disabled={isLoading || !input.trim()}
                  >
                    Send
                  </button>
                </form>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
