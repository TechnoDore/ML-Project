import React, { useState, useEffect, useRef } from 'react';
import { 
  Send, Download, Sparkles, 
  Moon, Sun, Calendar, BookOpen, MapPin, User,
  Maximize2, Minimize2, Search
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

// --- 1. TYPEWRITER EFFECT ---
const Typewriter = ({ text, speed = 20, onComplete }) => {
  const [displayedText, setDisplayedText] = useState('');
  const indexRef = useRef(0);
  const hasCompletedRef = useRef(false); 

  useEffect(() => {
    if (!text || hasCompletedRef.current) {
        if (text && hasCompletedRef.current) setDisplayedText(text);
        return;
    }

    setDisplayedText('');
    indexRef.current = 0;
    hasCompletedRef.current = false;
    
    const interval = setInterval(() => {
      if (indexRef.current < text.length) {
        setDisplayedText((prev) => prev + text.charAt(indexRef.current));
        indexRef.current++;
      } else {
        clearInterval(interval);
        hasCompletedRef.current = true;
        if (onComplete) onComplete();
      }
    }, speed);

    return () => clearInterval(interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, speed]); 

  const parts = displayedText.split(/(\*\*.*?\*\*)/g);
  return (
    <span className="whitespace-pre-wrap leading-relaxed">
      {parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={idx} className="font-bold text-blue-400 dark:text-blue-600">{part.slice(2, -2)}</strong>;
        }
        return part;
      })}
    </span>
  );
};

// --- 2. DYNAMIC ANIMATED BOT AVATAR ---
const AnimatedBot = ({ mood, category }) => {
  
  // Dynamic Color Logic based on Category
  const getColors = () => {
    if (mood === 'confused' || category === 'error') return { bg: 'bg-red-500', gradient: 'from-red-500 to-pink-600', shadow: 'shadow-red-500/50' };
    if (mood === 'thinking') return { bg: 'bg-purple-500', gradient: 'from-violet-500 to-fuchsia-500', shadow: 'shadow-purple-500/50' };
    
    switch (category) {
      case 'deadline':
        return { bg: 'bg-orange-500', gradient: 'from-orange-500 to-red-500', shadow: 'shadow-orange-500/50' };
      case 'education': // Syllabus, PYQ
        return { bg: 'bg-blue-500', gradient: 'from-blue-500 to-cyan-400', shadow: 'shadow-blue-500/50' };
      case 'person': // Professors
        return { bg: 'bg-emerald-500', gradient: 'from-emerald-500 to-teal-500', shadow: 'shadow-emerald-500/50' };
      case 'location':
        return { bg: 'bg-amber-400', gradient: 'from-amber-400 to-yellow-500', shadow: 'shadow-amber-500/50' };
      default:
        return { bg: 'bg-cyan-500', gradient: 'from-cyan-500 to-blue-600', shadow: 'shadow-cyan-500/50' };
    }
  };

  const colors = getColors();

  return (
    <div className="relative w-14 h-14 flex items-center justify-center transition-all duration-700">
      {/* Pulse Effect */}
      <div className={`absolute inset-0 rounded-full opacity-20 animate-ping ${colors.bg}`}></div>
      
      {/* Main Body */}
      <div className={`relative z-10 w-10 h-10 bg-gradient-to-br ${colors.gradient} rounded-xl shadow-lg ${colors.shadow} flex items-center justify-center transition-all duration-700 ${mood === 'thinking' ? 'animate-bounce' : 'animate-bounce-slow'}`}>
        
        {/* Eyes */}
        <div className={`flex space-x-1 transition-all duration-300 ${mood === 'confused' ? 'items-end mb-0.5' : 'items-center'}`}>
          <div className={`bg-white rounded-full transition-all ${mood === 'thinking' ? 'w-1.5 h-1.5 animate-pulse' : 'w-2 h-2 animate-blink'} ${mood === 'confused' ? 'w-1.5 h-1.5 mb-0.5' : ''}`}></div>
          <div className={`bg-white rounded-full transition-all ${mood === 'thinking' ? 'w-1.5 h-1.5 animate-pulse delay-75' : 'w-2 h-2 animate-blink delay-75'} ${mood === 'confused' ? 'w-2 h-2' : ''}`}></div>
        </div>
        
        {/* Mouth */}
        <div className={`absolute bottom-2 transition-all duration-300 
          ${mood === 'happy' ? 'w-3 h-1 border-b border-white/70 rounded-full' : ''} 
          ${mood === 'thinking' ? 'w-1.5 h-1.5 border border-white/70 rounded-full' : ''} 
          ${mood === 'confused' ? 'w-3 h-0.5 bg-white/70 rotate-[-10deg]' : ''}
        `}></div>
      </div>
    </div>
  );
};

// --- 3. MAIN CHAT COMPONENT ---
const UniversityChatbot = () => {
  const [messages, setMessages] = useState([
    { 
      type: 'bot', 
      text: "Hi! I'm **BU Buddy**. I can help you with Deadlines 📅, Professors 👤, and Campus Maps 📍. Try asking me something!", 
      timestamp: new Date().toISOString(),
      isNew: true 
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [botMood, setBotMood] = useState('happy');
  const [botCategory, setBotCategory] = useState('default'); // New State for Color
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [isFullScreen, setIsFullScreen] = useState(false); 
  
  const [filteredSuggestions, setFilteredSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const messagesEndRef = useRef(null);

  const ALL_SUGGESTIONS = [
    "Syllabus for CSET301",
    "Who teaches AI?",
    "Where is the Library?",
    "Upcoming deadlines",
    "Add deadline Project by 2025-12-01",
    "Mark task as done",
    "PYQ for Engineering Calculus",
    "Contact details for Prof. Priya",
    "Where does Dr. Vikram sit?",
    "List all professors",
    "Is there any work pending?",
    "Location of Cafeteria",
    "Show passed deadlines",
    "Question paper for Physics",
    "What will we learn in Web Security?"
  ];

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });

  useEffect(() => scrollToBottom(), [messages]);

  // --- HELPER: DETERMINE CATEGORY FROM TEXT ---
  const analyzeTextForCategory = (text) => {
    const lower = text.toLowerCase();
    
    // FIX: Check for greeting context first to prevent false positive on "Deadline" word
    if (lower.includes('bu buddy') || lower.includes('i can help with')) return 'default';

    if (lower.includes('deadline') || lower.includes('due') || lower.includes('task') || lower.includes('work') || lower.includes('mark') || lower.includes('pending')) return 'deadline';
    if (lower.includes('syllabus') || lower.includes('pdf') || lower.includes('paper') || lower.includes('pyq') || lower.includes('exam')) return 'education';
    if (lower.includes('professor') || lower.includes('email') || lower.includes('office') || lower.includes('dr.') || lower.includes('teach')) return 'person';
    if (lower.includes('location') || lower.includes('building') || lower.includes('floor') || lower.includes('where is')) return 'location';
    if (lower.includes('couldn\'t find') || lower.includes('sorry') || lower.includes('error')) return 'error';
    return 'default';
  };

  const quickActions = [
    { icon: <Calendar size={14} />, text: "Deadlines", query: "What are my upcoming deadlines?" },
    { icon: <BookOpen size={14} />, text: "Syllabus", query: "Syllabus for CSET301" },
    { icon: <MapPin size={14} />, text: "Library", query: "Where is the Library?" },
    { icon: <User size={14} />, text: "Professors", query: "List all professors" },
  ];

  const handleInputChange = (e) => {
    const userInput = e.target.value;
    setInput(userInput);

    // Real-time color changing while typing
    if (userInput.length > 2) {
        const likelyCategory = analyzeTextForCategory(userInput);
        if (likelyCategory !== 'default') setBotCategory(likelyCategory);
    }

    if (userInput.length > 0) {
      const filtered = ALL_SUGGESTIONS.filter(
        (suggestion) =>
          suggestion.toLowerCase().includes(userInput.toLowerCase())
      );
      setFilteredSuggestions(filtered);
      setShowSuggestions(true);
    } else {
      setShowSuggestions(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    setInput(suggestion);
    setShowSuggestions(false);
    // Immediately update color on suggestion click
    setBotCategory(analyzeTextForCategory(suggestion));
  };

  const handleSendMessage = async (textOverride = null) => {
    const textToSend = textOverride || input;
    if (!textToSend || !textToSend.trim() || isLoading) return;

    setShowSuggestions(false);
    
    // 1. Set Mood based on User Input immediately
    const inputCategory = analyzeTextForCategory(textToSend);
    setBotCategory(inputCategory);
    setBotMood('thinking');
    setIsLoading(true);

    const userMsg = { type: 'user', text: textToSend, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend })
      });

      if (!res.ok) throw new Error('Server Error');

      const data = await res.json();
      
      const responseText = data.response || "No response received.";
      
      // 2. Update Mood/Color based on Bot Response
      const responseCategory = analyzeTextForCategory(responseText);
      setBotCategory(responseCategory);

      const negativeKeywords = ["couldn't find", "sorry", "oops", "don't have info"];
      if (negativeKeywords.some(k => responseText.toLowerCase().includes(k))) {
        setBotMood('confused');
        setBotCategory('error');
        setTimeout(() => setBotMood('happy'), 3000);
      } else {
        setBotMood('happy');
      }

      const botMsg = {
        type: 'bot',
        text: responseText,
        timestamp: data.timestamp || new Date().toISOString(),
        data: data.data,
        isNew: true 
      };
      setMessages(prev => [...prev, botMsg]);

    } catch (error) {
      setMessages(prev => [...prev, { type: 'bot', text: "❌ Connection failed. Please check if the backend is running.", timestamp: new Date().toISOString(), isNew: true }]);
      setBotMood('confused');
      setBotCategory('error');
    } finally {
      setIsLoading(false);
    }
  };

  const MarkdownRenderer = ({ text }) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return (
      <p className="whitespace-pre-wrap text-sm leading-relaxed">
        {parts.map((part, index) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={index} className={`font-bold ${isDarkMode ? 'text-blue-300' : 'text-blue-600'}`}>{part.slice(2, -2)}</strong>;
          }
          return part;
        })}
      </p>
    );
  };

  return (
    <div className={`flex h-screen w-full font-sans transition-colors duration-300 overflow-hidden 
      ${isDarkMode ? 'bg-slate-900 text-gray-100' : 'bg-gray-100 text-slate-900'}
      ${isFullScreen ? 'p-0' : 'p-4 items-center justify-center'} 
    `}>
      
      {/* Background Ambiance - Changes with Category */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none transition-colors duration-1000">
         <div className={`absolute top-[-10%] right-[-5%] w-[500px] h-[500px] rounded-full blur-[100px] opacity-20 animate-pulse 
            ${botCategory === 'deadline' ? 'bg-red-600' : 
              botCategory === 'education' ? 'bg-blue-600' : 
              botCategory === 'person' ? 'bg-emerald-600' : 
              botCategory === 'location' ? 'bg-yellow-600' : 
              isDarkMode ? 'bg-blue-600' : 'bg-blue-300'}`}></div>
         
         <div className={`absolute bottom-[-10%] left-[-5%] w-[400px] h-[400px] rounded-full blur-[80px] opacity-20 animate-pulse delay-1000 
            ${botCategory === 'deadline' ? 'bg-orange-600' : 
              botCategory === 'education' ? 'bg-cyan-600' : 
              botCategory === 'person' ? 'bg-teal-600' : 
              botCategory === 'location' ? 'bg-amber-600' : 
              isDarkMode ? 'bg-purple-600' : 'bg-purple-300'}`}></div>
      </div>

      <div className={`relative z-10 flex flex-col overflow-hidden transition-all duration-500 ease-in-out
        ${isFullScreen 
          ? 'w-full h-full rounded-none border-0' 
          : 'h-full max-h-[90vh] w-full max-w-5xl rounded-3xl shadow-2xl border border-white/10'
        }
        ${isDarkMode ? 'bg-white/5 backdrop-blur-lg' : 'bg-white/80 backdrop-blur-md border-gray-200'}
      `}>
        
        <div className={`flex items-center justify-between p-4 md:p-6 border-b backdrop-blur-md z-20
          ${isDarkMode ? 'border-white/10 bg-black/20' : 'border-gray-200 bg-white/50'}
        `}>
          <div className="flex items-center gap-4">
            {/* Pass both mood and category to the Avatar */}
            <AnimatedBot mood={botMood} category={botCategory} />
            <div>
              <h2 className="text-xl font-bold">BU Buddy</h2>
              <div className="flex items-center gap-2">
                 <span className="relative flex h-2 w-2">
                  <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 
                      ${botCategory === 'deadline' ? 'bg-red-400' : 'bg-green-400'}`}></span>
                  <span className={`relative inline-flex rounded-full h-2 w-2 
                      ${botCategory === 'deadline' ? 'bg-red-500' : 'bg-green-500'}`}></span>
                </span>
                <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                   {isLoading ? 'Thinking...' : 'Online & Ready'}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <button 
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-2 rounded-lg transition-colors ${isDarkMode ? 'hover:bg-white/10 text-gray-300' : 'hover:bg-gray-100 text-gray-600'}`}
              title="Toggle Theme"
            >
              {isDarkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>

            <Sparkles className={`text-yellow-400 animate-pulse hidden sm:block`} size={20} />
            
            <button 
              onClick={() => setIsFullScreen(!isFullScreen)}
              className={`p-2 rounded-lg transition-colors ${isDarkMode ? 'hover:bg-white/10 text-gray-300' : 'hover:bg-gray-100 text-gray-600'}`}
              title={isFullScreen ? "Exit Full Screen" : "Full Screen"}
            >
              {isFullScreen ? <Minimize2 size={20} /> : <Maximize2 size={20} />}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 scrollbar-hide">
          {messages.map((msg, idx) => (
            <div key={idx} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} group animate-fade-in-up`}>
              
              {msg.type === 'bot' && (
                <div className={`mr-3 mt-1 hidden md:flex w-8 h-8 rounded-full items-center justify-center text-white text-xs font-bold shadow-md
                    ${botCategory === 'deadline' ? 'bg-gradient-to-br from-orange-500 to-red-600' : 
                      botCategory === 'education' ? 'bg-gradient-to-br from-blue-500 to-cyan-600' : 
                      'bg-gradient-to-br from-cyan-500 to-blue-600'}
                `}>
                    AI
                </div>
              )}

              <div className={`max-w-[85%] md:max-w-[75%] rounded-2xl px-5 py-4 shadow-sm transition-all hover:shadow-md ${
                msg.type === 'user' 
                  ? 'bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-tr-sm' 
                  : isDarkMode 
                    ? 'bg-slate-800/80 border border-white/5 text-gray-200 rounded-tl-sm backdrop-blur-sm' 
                    : 'bg-white border border-gray-100 text-gray-800 rounded-tl-sm'
              }`}>
                {msg.type === 'bot' && msg.isNew ? (
                  <Typewriter text={msg.text || ""} onComplete={() => {}} />
                ) : (
                   <MarkdownRenderer text={msg.text} />
                )}

                {msg.type === 'bot' && msg.data?.pdf_url && (
                  <a href={`${API_URL}/static/${msg.data.pdf_url}`} target="_blank" rel="noreferrer" 
                      className={`mt-4 flex items-center gap-3 rounded-xl px-4 py-3 text-xs font-bold transition-all border ${
                        isDarkMode 
                         ? 'bg-black/20 border-white/10 text-cyan-300 hover:bg-cyan-900/20' 
                         : 'bg-blue-50 border-blue-100 text-blue-600 hover:bg-blue-100'
                      }`}>
                    <Download size={16} />
                    Download Resource PDF
                  </a>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start animate-pulse">
               <div className="mr-3 w-8 h-8 rounded-full bg-gray-300/20 hidden md:block"></div>
               <div className={`rounded-2xl rounded-tl-sm px-5 py-4 flex space-x-1 items-center ${isDarkMode ? 'bg-slate-800/50' : 'bg-white'}`}>
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce delay-100"></div>
                  <div className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-bounce delay-200"></div>
               </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className={`p-4 md:p-6 border-t backdrop-blur-lg z-20 relative ${isDarkMode ? 'bg-slate-900/80 border-white/5' : 'bg-white/80 border-gray-100'}`}>
          
          {showSuggestions && filteredSuggestions.length > 0 && (
            <div className={`absolute bottom-full left-0 w-full mb-2 px-6 z-50`}>
              <div className={`rounded-2xl shadow-2xl border overflow-hidden max-h-48 overflow-y-auto ${
                isDarkMode 
                  ? 'bg-slate-800 border-white/10' 
                  : 'bg-white border-gray-200'
              }`}>
                {filteredSuggestions.map((suggestion, index) => (
                  <div
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className={`px-4 py-3 text-sm cursor-pointer transition-colors flex items-center gap-2 ${
                      isDarkMode 
                        ? 'text-gray-200 hover:bg-white/10' 
                        : 'text-gray-700 hover:bg-blue-50'
                    }`}
                  >
                    <Search size={14} className="opacity-50" />
                    {suggestion}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex gap-2 overflow-x-auto pb-3 scrollbar-hide mb-2">
            {quickActions.map((action, i) => (
              <button
                key={i}
                onClick={() => handleSendMessage(action.query)}
                className={`flex-shrink-0 flex items-center gap-2 px-4 py-2 rounded-full text-xs font-medium transition-all transform hover:scale-105 active:scale-95 border ${
                  isDarkMode 
                    ? 'bg-slate-800 border-white/10 hover:bg-slate-700 text-blue-300' 
                    : 'bg-white border-gray-200 hover:bg-gray-50 text-blue-600 shadow-sm'
                }`}
              >
                {action.icon} {action.text}
              </button>
            ))}
          </div>

          <div className="relative flex items-center gap-2">
            <div className={`flex-1 flex items-center rounded-2xl border px-4 py-3 transition-all focus-within:ring-2 focus-within:ring-blue-500/50 ${
              isDarkMode 
                ? 'bg-slate-800 border-white/10 focus-within:border-blue-500/50' 
                : 'bg-white border-gray-200 shadow-sm focus-within:border-blue-400'
            }`}>
              <input
                type="text"
                value={input}
                onChange={handleInputChange} 
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask about deadlines, courses, or locations..."
                className="flex-1 bg-transparent outline-none text-sm placeholder-gray-400 min-w-0"
                autoComplete="off"
              />
            </div>
            
            <button
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || isLoading}
              className={`flex-shrink-0 w-12 h-12 flex items-center justify-center rounded-2xl text-white shadow-lg hover:scale-105 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed transition-all 
                ${botCategory === 'deadline' ? 'bg-gradient-to-br from-orange-500 to-red-600 shadow-red-500/30' : 
                  botCategory === 'education' ? 'bg-gradient-to-br from-blue-500 to-cyan-600 shadow-blue-500/30' : 
                  'bg-gradient-to-br from-cyan-500 to-blue-600 shadow-cyan-500/30'}
              `}
            >
              <Send size={20} className={isLoading ? 'hidden' : 'block'} />
              {isLoading && <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>}
            </button>
          </div>
        </div>

      </div>

      <style>{`
        @keyframes bounce-slow {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
        @keyframes blink {
          0%, 90%, 100% { transform: scaleY(1); }
          95% { transform: scaleY(0.1); }
        }
        @keyframes fade-in-up {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-bounce-slow { animation: bounce-slow 3s infinite ease-in-out; }
        .animate-blink { animation: blink 4s infinite; }
        .animate-fade-in-up { animation: fade-in-up 0.3s ease-out forwards; }
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
};

export default UniversityChatbot;