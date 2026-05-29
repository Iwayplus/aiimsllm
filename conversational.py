"""
Conversational AI Module for AIIMS Jammu Hospital Chatbot
Handles all small talk, conversational intents, and natural conversation flow
"""

import logging
import random
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from rapidfuzz import fuzz
from langchain_groq import ChatGroq
from zoneinfo import ZoneInfo

# Configure logging
logger = logging.getLogger(__name__)

class ConversationalEngine:
    """Enhanced conversational engine for natural dialogue"""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key
        self.conversation_patterns = self._initialize_patterns()
        self.response_templates = self._initialize_templates()
        
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize conversation patterns for intent detection"""
        return {
            'greeting': {
                'exact_matches': [
                    'hi', 'hello', 'hey', 'hiya', 'yo', 'sup', 'howdy', 'greetings',
                    'good morning', 'good afternoon', 'good evening', 'good night',
                    'morning', 'afternoon', 'evening', 'namaste', 'namaskar',
                    'salaam', 'adaab', 'pranam', 'vanakkam'
                ],
                'partial_patterns': [
                    r'\b(hi|hello|hey)\s+(there|aiims|bot|assistant)\b',
                    r'\b(good\s+(morning|afternoon|evening|night))\b',
                    r'\b(hey\s+(what\'?s\s+up|whats\s+up|wassup))\b',
                    r'^\s*(hi|hello|hey)\s*[.!]*\s*$'
                ],
                'fuzzy_threshold': 90
            },
            
            'farewell': {
                'exact_matches': [
                    'bye', 'goodbye', 'bye bye', 'see you', 'see ya', 'cya', 'later',
                    'take care', 'farewell', 'adios', 'tata', 'alvida', 'khuda hafiz',
                    'gotta go', 'gtg', 'leaving', 'peace out', 'catch you later',
                    'talk to you later', 'ttyl', 'until next time', 'signing off'
                ],
                'partial_patterns': [
                    r'\b(bye|goodbye)\s+(for\s+now|then|bye)\b',
                    r'\b(see\s+you\s+(later|soon|tomorrow|next\s+time))\b',
                    r'\b(take\s+care|be\s+safe|stay\s+safe)\b',
                    r'\b(i\s+(gotta|have\s+to|need\s+to)\s+go)\b',
                    r'\b(thanks?\s+(and\s+)?(bye|goodbye))\b'
                ],
                'fuzzy_threshold': 85
            },
            
            'gratitude': {
                'exact_matches': [
                    'thanks', 'thank you', 'thx', 'ty', 'tysm', 'thank u',
                    'appreciate it', 'much appreciated', 'grateful', 'thankful',
                    'dhanyawad', 'shukriya', 'meherbani'
                ],
                'partial_patterns': [
                    r'\b(thanks?\s+(a\s+lot|so\s+much|very\s+much))\b',
                    r'\b(thank\s+you\s+(so\s+much|very\s+much|again))\b',
                    r'\b(really\s+appreciate\s+(it|this|that))\b',
                    r'\b(you\'?re\s+(awesome|great|amazing|helpful))\b',
                    r'\b(that\'?s\s+(helpful|useful|great|perfect))\b'
                ],
                'fuzzy_threshold': 88
            },
            
            'smalltalk': {
                'exact_matches': [
                    'how are you', 'whats up', 'wassup', 'sup', 'how you doing',
                    'how have you been', 'bored', 'nothing much', 'just checking',
                    'tell me something', 'say something', 'you there', 'you awake',
                    'anyone there', 'hello there', 'testing', 'test'
                ],
                'partial_patterns': [
                    r'\b(how\s+(are\s+you|you\s+doing|things|life))\b',
                    r'\b(what\'?s\s+(up|new|happening|going\s+on))\b',
                    r'\b(tell\s+me\s+(something|anything)\s+(interesting|new|good))\b',
                    r'\b(i\'?m\s+(bored|tired|back|here\s+again))\b',
                    r'\b(just\s+(saying\s+hi|checking\s+in|curious))\b',
                    r'\b(you\s+(online|available|working|active))\b'
                ],
                'fuzzy_threshold': 82
            },
            
            'confirmation': {
                'exact_matches': [
                    'yes', 'yeah', 'yep', 'yup', 'sure', 'ok', 'okay', 'alright',
                    'correct', 'right', 'exactly', 'absolutely', 'definitely',
                    'of course', 'indeed', 'affirmative', 'true', 'agreed',
                    'haan', 'ji haan', 'bilkul', 'sahi'
                ],
                'partial_patterns': [
                    r'\b(yes\s+(please|sure|of\s+course))\b',
                    r'\b(that\'?s\s+(right|correct|true|perfect))\b',
                    r'\b(you\'?re\s+(right|correct))\b',
                    r'\b(i\s+agree)\b'
                ],
                'fuzzy_threshold': 95
            },
            
            'negation': {
                'exact_matches': [
                    'no', 'nope', 'nah', 'never', 'not really', 'dont think so',
                    "don't think so", 'wrong', 'incorrect', 'false', 'negative',
                    'nahi', 'nahin', 'bilkul nahi'
                ],
                'partial_patterns': [
                    r'\b(no\s+(way|thanks|thank\s+you))\b',
                    r'\b(that\'?s\s+(wrong|incorrect|not\s+right))\b',
                    r'\b(i\s+don\'?t\s+(think\s+so|agree))\b',
                    r'\b(not\s+(really|exactly|quite))\b'
                ],
                'fuzzy_threshold': 90
            },
            
            'help_request': {
                'exact_matches': [
                    'help', 'help me', 'can you help', 'need help', 'assist me',
                    'support', 'guide me', 'what can you do', 'how can you help'
                ],
                'partial_patterns': [
                    r'\b(can\s+you\s+(help|assist|support|guide)\s+me)\b',
                    r'\b(i\s+need\s+(help|assistance|support|guidance))\b',
                    r'\b(could\s+you\s+(help|assist))\b',
                    r'\b(please\s+(help|assist))\b',
                    r'\b(what\s+can\s+you\s+(do|help\s+with))\b'
                ],
                'fuzzy_threshold': 85
            },
            
            'mood_expression': {
                'exact_matches': [
                    'feeling good', 'feeling bad', 'happy', 'sad', 'excited',
                    'worried', 'anxious', 'nervous', 'stressed', 'relieved',
                    'confused', 'frustrated', 'angry', 'disappointed'
                ],
                'partial_patterns': [
                    r'\b(i\'?m\s+(feeling|so)\s+(good|bad|happy|sad|worried|nervous|stressed|excited))\b',
                    r'\b(feeling\s+(a\s+bit|quite|very)\s+(good|bad|worried|nervous))\b',
                    r'\b(having\s+a\s+(good|bad|tough|hard)\s+day)\b',
                    r'\b(bit\s+(worried|nervous|scared|anxious))\b'
                ],
                'fuzzy_threshold': 80
            },
            
            'compliment': {
                'exact_matches': [
                    'good job', 'well done', 'great work', 'excellent', 'amazing',
                    'awesome', 'brilliant', 'fantastic', 'superb', 'wonderful',
                    'impressive', 'outstanding', 'perfect', 'spot on'
                ],
                'partial_patterns': [
                    r'\b(you\'?re\s+(amazing|awesome|great|brilliant|helpful|smart))\b',
                    r'\b(that\'?s\s+(amazing|awesome|great|brilliant|perfect|excellent))\b',
                    r'\b(really\s+(good|great|helpful|impressive))\b',
                    r'\b(love\s+(it|this|that|your\s+response))\b'
                ],
                'fuzzy_threshold': 85
            }
        }
    
    def _initialize_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates for different intents"""
        return {
            'greeting': [
                "{time_greeting}! 😊 {user_context} How can I help you today?",
                "{time_greeting}! {returning_user_text} Ready to assist with any hospital questions 👋",
                "{time_greeting}! {mood_text} How can I make your AIIMS visit easier today? 😊"
            ],
            'farewell': [
                "Take care! Hope your visit goes smoothly. I'm here whenever you need help! 👋",
                "Goodbye! Wishing you all the best with your hospital visit. Feel free to reach out anytime! 🙏",
                "See you later! Hope everything works out well for you at AIIMS Jammu! 👋"
            ],
            'gratitude': [
                "You're so welcome! 😊 Always happy to help at AIIMS Jammu!",
                "Glad I could help{topic_reference}! Anything else you need? 😊",
                "My pleasure! That's what I'm here for. Need anything else? 😊"
            ],
            'smalltalk': [
                "{smalltalk_response} What brings you to AIIMS today? 😊",
                "{smalltalk_response} How can I assist with your hospital needs? 😊",
                "{smalltalk_response} Any questions about AIIMS Jammu? 😊"
            ],
            'confirmation': [
                "Great! {confirmation_context} Is there anything else I can help you with?",
                "Perfect! {confirmation_context} What else would you like to know?",
                "Excellent! {confirmation_context} How else can I assist you today? 😊"
            ],
            'negation': [
                "No problem! {negation_context} What would you like to know instead?",
                "That's alright! {negation_context} How else can I help you?",
                "Understood! {negation_context} Is there something else I can assist with? 😊"
            ],
            'help_request': [
                "Absolutely! I can help with directions, doctor info, appointments, services - whatever you need at AIIMS Jammu. What would you like to know?",
                "Of course! I'm here to assist with anything hospital-related. What do you need help with?",
                "Happy to help! Whether it's finding doctors, departments, or services - just let me know what you need! 😊"
            ],
            'mood_expression': [
                "{mood_acknowledgment} How can I help today?",
                "{empathy_response} What can I assist you with at AIIMS?",
                "{supportive_response} How can I make your visit easier? 😊"
            ],
            'compliment': [
                "Thank you so much! 😊 That really means a lot. How else can I help you today?",
                "You're too kind! 🙏 I'm just here to make your AIIMS experience better. What else do you need?",
                "Aww, thank you! 😊 I'm glad I could help. Is there anything else you'd like to know?"
            ]
        }

    def detect_conversational_intent(self, query: str) -> Optional[str]:
        """Enhanced conversational intent detection with better pattern matching"""
        query_clean = query.lower().strip()
        query_normalized = re.sub(r'[^\w\s]', '', query_clean)
        
        def fuzzy_match_with_context(text: str, patterns: Dict, threshold: int = 85) -> Tuple[bool, int]:
            """Enhanced fuzzy matching with context awareness"""
            text_clean = text.lower().strip()
            
            # Check exact matches first
            for pattern in patterns.get('exact_matches', []):
                if pattern == text_clean:
                    return True, 100
                if len(text_clean.split()) <= 3 and pattern in text_clean:
                    return True, 95
            
            # Check partial patterns with regex
            for pattern in patterns.get('partial_patterns', []):
                if re.search(pattern, text_clean):
                    return True, 90
            
            # Fuzzy matching
            best_score = 0
            for pattern in patterns.get('exact_matches', []):
                score = fuzz.ratio(text_clean, pattern)
                if score > best_score:
                    best_score = score
            
            threshold = patterns.get('fuzzy_threshold', threshold)
            return best_score >= threshold, best_score
        
        # Check each intent
        intent_scores = {}
        for intent, patterns in self.conversation_patterns.items():
            is_match, score = fuzzy_match_with_context(query_clean, patterns)
            if is_match:
                intent_scores[intent] = score
        
        # Handle mixed intents
        if len(intent_scores) > 1:
            if 'gratitude' in intent_scores and 'farewell' in intent_scores:
                return 'grateful_farewell'
            elif 'gratitude' in intent_scores and 'compliment' in intent_scores:
                return 'grateful_compliment'
            elif 'greeting' in intent_scores and 'help_request' in intent_scores:
                return 'greeting_with_help'
            elif 'smalltalk' in intent_scores and 'mood_expression' in intent_scores:
                return 'mood_smalltalk'
            elif 'confirmation' in intent_scores and any(k in intent_scores for k in ['gratitude', 'compliment']):
                return 'positive_confirmation'
            elif 'negation' in intent_scores and 'help_request' in intent_scores:
                return 'corrective_help'
        
        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Edge cases
        if len(query_clean.split()) <= 2:
            if query_clean in ['k', 'ok', 'hmm', 'uh', 'um', 'ah']:
                return 'minimal_response'
            elif '?' in query:
                return 'question'
        
        return None

    def get_time_context(self) -> Tuple[str, str]:
        """Get time-based greeting and context based on IST timezone"""
        try:
            ist_time = datetime.now(ZoneInfo("Asia/Kolkata"))
        except Exception as e:
            logging.error(f"[get_time_context] Failed to get IST time: {e}")
            ist_time = datetime.utcnow()  # fallback

        logging.info(f"[get_time_context] IST Time Used: {ist_time.strftime('%Y-%m-%d %H:%M:%S')}")
        current_hour = ist_time.hour
        
        if 5 <= current_hour < 12:
            return random.choice(["Good morning", "Morning"]), "morning"
        elif 12 <= current_hour < 17:
            return random.choice(["Good afternoon", "Afternoon"]), "afternoon"
        elif 17 <= current_hour < 21:
            return random.choice(["Good evening", "Evening"]), "evening"
        else:
            return random.choice(["Good evening", "Hello"]), "night"

    def analyze_user_context(self, memory, user_query: str) -> Dict[str, Any]:
        """Analyze user context from memory"""
        context = {
            'returning_user': False,
            'conversation_flow': 'new',
            'user_mood': 'neutral',
            'recent_topics': [],
            'preferences': {},
            'conversation_style': {},
            'last_intent': None,
            'awaiting_confirmation': False
        }
        
        try:
            if hasattr(memory, 'get_relevant_context'):
                # New memory system
                context_data = memory.get_relevant_context(user_query, max_items=5)
                recent_history = context_data.get('recent_history', [])
                
                if recent_history:
                    context['returning_user'] = True
                    
                    # Get last interection intent
                    last_turn = recent_history[-1] if recent_history else {}
                    if last_turn and 'entities' in last_turn:
                        interaction_types = last_turn['entities'].get('interaction_types', [])
                        if interaction_types:
                            context['last_intent'] = interaction_types[0]

                    # Check if we're awaiting user confirmation
                    if context['last_intent'] in ['help_request', 'greeting_with_help']:
                        context['awaiting_confirmation'] = True

                    # Determine conversation flow
                    if last_turn:
                        current_time = time.time()
                        time_since_last = current_time - last_turn.get('timestamp', 0)
                        if time_since_last < 300:  # 5 minutes
                            context['conversation_flow'] = "continuing"
                        elif time_since_last < 3600:  # 1 hour
                            context['conversation_flow'] = "resuming"
                        else:
                            context['conversation_flow'] = "returning"
                
                # Extract user insights
                user_profile = context_data.get('user_profile', {})
                context.update({
                    'recent_topics': [t['name'] for t in context_data.get('current_topics', [])[:3]],
                    'preferences': user_profile.get('preferences', {}),
                    'conversation_style': user_profile.get('conversation_style', {})
                })
                
                # Detect mood from recent interactions
                recent_sentiments = [turn.get('sentiment', 'neutral') for turn in recent_history[-3:]]
                if 'positive' in recent_sentiments or 'very_positive' in recent_sentiments:
                    context['user_mood'] = "positive"
                elif 'negative' in recent_sentiments or 'very_negative' in recent_sentiments:
                    context['user_mood'] = "concerned"
                    
            elif hasattr(memory, 'short_term_history'):
                # Old memory system
                if memory.short_term_history:
                    context['returning_user'] = True
                    context['conversation_flow'] = "continuing"
                    
        except Exception as e:
            logger.warning(f"Could not analyze user context: {e}")
        
        return context

    def generate_llm_response(self, user_query: str, context: Dict[str, Any], intent: str) -> str:
        """Generate response using LLM if available, otherwise use templates"""
        
        if not self.groq_api_key:
            return self._generate_template_response(user_query, context, intent)
        
        # Build enhanced prompt
        time_greeting, time_context = self.get_time_context()

        # Add context for confirmation/negation handling
        context_info = ""
        if intent in ['confirmation', 'negation'] and context.get('last_intent'):
            context_info = f"- Previous Intent: {context['last_intent']}\n- User is responding to previous interaction"
        
        conversation_context = f"""
You are AiimsBot, a warm, empathetic, and naturally conversational AI assistant at AIIMS Jammu hospital.

CURRENT CONTEXT:
- Time: {time_context}
- User Status: {'Returning visitor' if context('returning_user') else 'New visitor'}
- Conversation Flow: {context('conversation_flow', 'unknown')}
- User Mood: {context('user_mood', 'neutral')}
- Intent Detected: {intent}
{context_info}

USER CONTEXT:
- Recent Topics: {', '.join(context.get('recent_topics', [])) or 'None'}
- Conversation Style: {context('conversation_style', {}).get('engagement_level', 'unknown')} engagement
- Awaiting Confirmation: {context.get('awaiting_confirmation', False)}

PERSONALITY GUIDELINES:
- Be genuinely warm and empathetic (hospital visitors may be stressed)
- Show personality while remaining professional
- Use conversational connectors and natural speech patterns
- Be concise but not robotic (15-25 words typically)
- Use appropriate emojis sparingly (0-2 max)
- Show genuine interest in helping

USER'S MESSAGE: "{user_query}"

Generate a natural, conversational response (just the response text, no labels):
"""
        
        try:
            llm = ChatGroq(api_key=self.groq_api_key, model="llama3-8b-8192", temperature=0.7)
            response = llm.invoke(conversation_context).content.strip()
            response = re.sub(r'^(AiimsBot:|Assistant:|Bot:)\s*', '', response)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_template_response(user_query, context, intent)

    def _generate_template_response(self, user_query: str, context: Dict[str, Any], intent: str) -> str:
        """Generate response using templates as fallback"""
        time_greeting, _ = self.get_time_context()
        query_lower = user_query.lower()
        
        # Template variables
        template_vars = {
            'time_greeting': time_greeting,
            'user_context': 'Great to see you again' if context['returning_user'] else 'Welcome to AIIMS Jammu',
            'returning_user_text': 'Back for more info?' if context['returning_user'] else 'Ready to assist',
            'mood_text': "Hope you're feeling alright" if context['user_mood'] == 'concerned' else '',
            'topic_reference': f" with {context['recent_topics'][0]}" if context['recent_topics'] else '',
            'smalltalk_response': "Doing great, thanks for asking!" if 'how are you' in query_lower else "I'm here and ready to help!",
            'mood_acknowledgment': "I understand - hospital visits can be stressful" if context['user_mood'] == 'concerned' else "I'm glad to hear that!",
            'empathy_response': "I'm here to make things easier for you" if context['user_mood'] == 'concerned' else "Ready to assist you",
            'supportive_response': "I understand" if context['user_mood'] == 'concerned' else "I'm here to help",
            'confirmation_context': self._get_confirmation_context(context),
            'negation_context': self._get_negation_context(context)
        }

        # Handle mixed intents
        if intent in ['grateful_farewell', 'grateful_compliment', 'greeting_with_help', 'mood_smalltalk', 
                     'positive_confirmation', 'corrective_help']:
            return self._handle_mixed_intent(intent, template_vars, context)
        
        # Get appropriate template
        templates = self.response_templates.get(intent, self.response_templates['greeting'])
        template = random.choice(templates)
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            return f"{time_greeting}! How can I help you at AIIMS Jammu today? 😊"
        
    def _get_confirmation_context(self, context: Dict[str, Any]) -> str:
        """Get context for confirmation responses"""
        last_intent = context.get('last_intent')
        if last_intent == 'help_request':
            return "I'm glad you'd like my assistance!"
        elif last_intent in ['greeting', 'greeting_with_help']:
            return "Let's get you the information you need!"
        elif context.get('recent_topics'):
            return f"Happy to continue helping with {context['recent_topics'][0]}!"
        return "I'm here to help!"

    def _get_negation_context(self, context: Dict[str, Any]) -> str:
        """Get context for negation responses"""
        last_intent = context.get('last_intent')
        if last_intent == 'help_request':
            return "That's perfectly fine!"
        elif context.get('recent_topics'):
            return f"No worries about {context['recent_topics'][0]}!"
        return "No problem at all!"
    
    def _handle_mixed_intent(self, intent: str, template_vars: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Handle mixed intents with appropriate responses"""
        mixed_responses = {
            'grateful_farewell': [
                f"You're very welcome! {template_vars['time_greeting']} and take care! 👋",
                "My pleasure! Hope your visit goes well. See you next time! 🙏",
                "Glad I could help! Take care and feel free to reach out anytime! 👋"
            ],
            'grateful_compliment': [
                "Thank you so much! 😊 That really makes my day. Anything else I can help with?",
                "You're too kind! 🙏 I'm just happy I could help. What else do you need?",
                "Aww, thank you! 😊 Your kind words mean a lot. How else can I assist?"
            ],
            'greeting_with_help': [
                f"{template_vars['time_greeting']}! I'd be happy to help with anything you need at AIIMS Jammu! 😊",
                f"{template_vars['time_greeting']}! Ready to assist with doctors, appointments, directions - whatever you need!",
                f"{template_vars['time_greeting']}! What can I help you with today? 😊"
            ],
            'mood_smalltalk': [
                f"{template_vars['smalltalk_response']} {template_vars['mood_acknowledgment']} What brings you to AIIMS?",
                f"{template_vars['smalltalk_response']} How can I make your hospital visit easier? 😊",
                f"{template_vars['empathy_response']} What do you need help with today?"
            ],
            'positive_confirmation': [
                f"Excellent! {template_vars['confirmation_context']} Thank you for confirming! 😊",
                f"Perfect! {template_vars['confirmation_context']} I appreciate that!",
                f"Great! {template_vars['confirmation_context']} You're awesome! 😊"
            ],
            'corrective_help': [
                f"No worries! {template_vars['negation_context']} Let me help you find what you're actually looking for.",
                f"That's alright! {template_vars['negation_context']} What would you like to know instead?",
                f"No problem! {template_vars['negation_context']} How can I better assist you? 😊"
            ]
        }
        
        responses = mixed_responses.get(intent, mixed_responses['greeting_with_help'])
        return random.choice(responses)

    def handle_small_talk(self, user_query: str, memory, user_id: str) -> Dict[str, str]:
        """Main function to handle small talk and conversational interactions"""
        
        # Detect intent
        intent = self.detect_conversational_intent(user_query)
        if not intent:
            return None  # Not a conversational query
        
        # Analyze user context
        context = self.analyze_user_context(memory, user_query)
        
        # Generate response
        response = self.generate_llm_response(user_query, context, intent)
        
        # Store interaction in memory if memory system is available
        self._store_interaction(user_query, response, intent, context, memory, user_id)
        
        return {"answer": response}

    def _store_interaction(self, user_query: str, response: str, intent: str, 
                          context: Dict[str, Any], memory, user_id: str):
        """Store the conversational interaction in memory"""
        try:
            # Determine sentiment
            positive_indicators = ['thanks', 'good', 'great', 'hello', 'hi', 'happy', 'excited', 
                                 'awesome', 'brilliant', 'perfect', 'excellent', 'yes', 'sure', 'absolutely']
            negative_indicators = ['worried', 'nervous', 'bad', 'sad', 'stressed', 'anxious', 
                                 'no', 'wrong', 'frustrated', 'disappointed', 'angry']
            query_lower = user_query.lower()
            
            if intent in ['compliment', 'gratitude', 'confirmation', 'positive_confirmation']:
                sentiment = "very_positive"
            elif intent in ['negation', 'corrective_help']:
                sentiment = "neutral"  # Not necessarily negative
            elif intent in ['mood_expression'] and any(word in query_lower for word in negative_indicators):
                sentiment = "concerned"
            elif any(word in query_lower for word in positive_indicators):
                sentiment = "positive"
            elif any(word in query_lower for word in negative_indicators):
                sentiment = "concerned"
            else:
                sentiment = "neutral"
            
            # Create entities
            extracted_entities = {
                'interaction_type': [intent],
                'user_mood': [context['user_mood']] if context['user_mood'] != 'neutral' else [],
                'conversation_flow': [context['conversation_flow']],
                'response_type': ['confirmation'] if intent in ['confirmation', 'positive_confirmation'] else
                                ['negation'] if intent in ['negation', 'corrective_help'] else
                                ['compliment'] if intent in ['compliment', 'grateful_compliment'] else
                                ['mixed'] if '_' in intent else ['simple']
            }

            # specific entities based on intent
            if intent in ['confirmation', 'positive_confirmation']:
                extracted_entities['user_agreement'] = ['yes']
            elif intent in ['negation', 'corrective_help']:
                extracted_entities['user_agreement'] = ['no']
            elif intent in ['compliment', 'grateful_compliment']:
                extracted_entities['user_satisfaction'] = ['high']
            elif intent == 'gratitude':
                extracted_entities['user_satisfaction'] = ['satisfied']
            
            # Determine importance
            importance_mapping = {
                'greeting': 0.6, 'farewell': 0.5, 'gratitude': 0.7,
                'mood_expression': 0.7, 'help_request': 0.8, 'smalltalk': 0.3,
                'confirmation': 0.8, 'negation': 0.7, 'compliment': 0.8,  # Higher importance for interaction intents
                'grateful_farewell': 0.6, 'grateful_compliment': 0.8,
                'greeting_with_help': 0.9, 'mood_smalltalk': 0.6,
                'positive_confirmation': 0.8, 'corrective_help': 0.7
            }
            importance = importance_mapping.get(intent, 0.4)
            
            # Create topics
            topics = ['small_talk', intent]

            if context['user_mood'] == "concerned":
                topics.append('user_concern')
            if context['user_mood'] == "positive":
                topics.append('positive_interaction')
            if context['returning_user']:
                topics.append('returning_visitor')
            if intent in ['confirmation', 'positive_confirmation']:
                topics.extend(['user_agreement', 'positive_feedback'])
            elif intent in ['negation', 'corrective_help']:
                topics.extend(['user_disagreement', 'course_correction'])
            elif intent in ['compliment', 'grateful_compliment']:
                topics.extend(['user_satisfaction', 'positive_feedback'])
            elif '_' in intent:  # Mixed intents
                topics.append('complex_interaction')
            
            # Add to memory
            if hasattr(memory, 'add_turn'):
                memory.add_turn(
                    user_query=user_query,
                    assistant_response=response,
                    extracted_entities=extracted_entities,
                    sentiment=sentiment,
                    importance=importance,
                    topics=topics
                )
            
        except Exception as e:
            logger.error(f"Error storing conversational interaction: {e}")

    def analyze_conversation_pattern(self, user_query: str, memory) -> Dict[str, Any]:
        """Analyze conversation patterns for improved responses"""
        pattern_analysis = {
            'query_length': len(user_query.split()),
            'question_type': 'none',
            'formality': 'casual',
            'urgency': 'normal',
            'complexity': 'simple',
            'emotional_tone': 'neutral',
            'interaction_style': 'standard'
        }
        
        query_lower = user_query.lower()
        
        # Question type detection
        if '?' in user_query:
            if query_lower.startswith(('what', 'how', 'where', 'when', 'why', 'who')):
                pattern_analysis['question_type'] = 'wh_question'
            elif query_lower.startswith(('can', 'could', 'would', 'should', 'is', 'are', 'do', 'does')):
                pattern_analysis['question_type'] = 'yes_no_question'
            else:
                pattern_analysis['question_type'] = 'other_question'
        
        # Formality detection
        formal_indicators = ['please', 'could you', 'would you', 'may i', 'excuse me', 'kindly']
        casual_indicators = ['hey', 'yo', 'sup', 'gonna', 'wanna', 'gotta', 'yeah', 'yep']
        
        if any(indicator in query_lower for indicator in formal_indicators):
            pattern_analysis['formality'] = 'formal'
        elif any(indicator in query_lower for indicator in casual_indicators):
            pattern_analysis['formality'] = 'very_casual'
        
        # Urgency detection
        urgent_indicators = ['urgent', 'emergency', 'quickly', 'asap', 'immediate', 'now', 'right now']
        if any(indicator in query_lower for indicator in urgent_indicators):
            pattern_analysis['urgency'] = 'high'
        
        # Complexity detection
        if len(user_query.split()) > 15 or query_lower.count('and') > 2:
            pattern_analysis['complexity'] = 'complex'
        elif len(user_query.split()) > 8:
            pattern_analysis['complexity'] = 'medium'

        # Emotional tone detection
        positive_emotions = ['happy', 'excited', 'great', 'awesome', 'love', 'amazing', 'brilliant']
        negative_emotions = ['worried', 'scared', 'nervous', 'frustrated', 'angry', 'sad', 'upset']
        
        if any(emotion in query_lower for emotion in positive_emotions):
            pattern_analysis['emotional_tone'] = 'positive'
        elif any(emotion in query_lower for emotion in negative_emotions):
            pattern_analysis['emotional_tone'] = 'negative'
        
        # Interaction style detection
        if any(word in query_lower for word in ['thanks', 'thank you', 'appreciate']):
            pattern_analysis['interaction_style'] = 'grateful'
        elif any(word in query_lower for word in ['yes', 'no', 'ok', 'sure']):
            pattern_analysis['interaction_style'] = 'responsive'
        elif any(word in query_lower for word in ['awesome', 'great', 'brilliant', 'perfect']):
            pattern_analysis['interaction_style'] = 'complimentary'
        
        return pattern_analysis
    
    def get_contextual_follow_up(self, intent: str, context: Dict[str, Any]) -> Optional[str]:
        """Generate contextual follow-up questions based on intent and context"""
        follow_ups = {
            'confirmation': [
                "What specific information are you looking for?",
                "Which department or service interests you?",
                "How can I make your visit smoother?"
            ],
            'negation': [
                "What would you like to know instead?",
                "Is there a different way I can help?",
                "What brings you to AIIMS today?"
            ],
            'compliment': [
                "Is there anything specific you'd like help with?",
                "What other questions do you have about AIIMS?",
                "How else can I assist your visit?"
            ],
            'gratitude': [
                "Is there anything else you need to know?",
                "Any other questions about your visit?",
                "What else can I help you with?"
            ]
        }
        
        # Don't provide follow-ups for farewell intents
        if intent in ['farewell', 'grateful_farewell']:
            return None
        
        # Get appropriate follow-ups
        intent_follow_ups = follow_ups.get(intent, follow_ups['confirmation'])
        
        # Customize based on context
        if context.get('user_mood') == 'concerned':
            return "How can I help ease your concerns about your visit?"
        elif context.get('recent_topics'):
            topic = context['recent_topics'][0]
            return f"Any other questions about {topic}?"
        
        return random.choice(intent_follow_ups)

    def should_escalate_to_specialist(self, intent: str, user_query: str, context: Dict[str, Any]) -> bool:
        """Determine if the conversation should be escalated to a specialist or medical query handler"""
        
        # Don't escalate pure conversational intents
        pure_conversational = ['greeting', 'farewell', 'gratitude', 'smalltalk', 'confirmation', 
                              'negation', 'compliment', 'mood_expression']
        
        if intent in pure_conversational:
            return False
        
        # Check for medical keywords that might need specialist handling
        medical_keywords = ['pain', 'symptoms', 'diagnosis', 'treatment', 'medicine', 'surgery', 
                           'appointment', 'doctor', 'specialist', 'emergency']
        
        query_lower = user_query.lower()
        if any(keyword in query_lower for keyword in medical_keywords):
            return True
        
        # Check context for medical topics
        if context.get('recent_topics'):
            medical_topics = ['appointment', 'doctor', 'department', 'medical', 'treatment']
            if any(topic in context['recent_topics'] for topic in medical_topics):
                return True
        
        return False


# Create global instance (will be imported in chat.py)
def create_conversational_engine(groq_api_key: Optional[str] = None) -> ConversationalEngine:
    """Factory function to create conversational engine"""
    return ConversationalEngine(groq_api_key)


# Backward compatibility functions for direct import
def detect_conversational_intent(query: str) -> Optional[str]:
    """Backward compatibility function"""
    engine = ConversationalEngine()
    return engine.detect_conversational_intent(query)


def handle_small_talk(user_query: str, memory, user_id: str, groq_api_key: Optional[str] = None) -> Dict[str, str]:
    """Backward compatibility function"""
    engine = ConversationalEngine(groq_api_key)
    return engine.handle_small_talk(user_query, memory, user_id)