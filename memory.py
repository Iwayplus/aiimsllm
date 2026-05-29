import threading
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import hashlib
import pickle
import os
from dataclasses import dataclass, asdict
from enum import Enum

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class EntityType(Enum):
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    EVENT = "event"
    CONCEPT = "concept"
    PREFERENCE = "preference"
    SKILL = "skill"
    GOAL = "goal"
    PROBLEM = "problem"
    SOLUTION = "solution"

@dataclass
class MemoryItem:
    content: str
    memory_type: MemoryType
    timestamp: float
    importance: float = 0.5  # 0.0 to 1.0
    access_count: int = 0
    last_accessed: float = 0.0
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed == 0.0:
            self.last_accessed = self.timestamp

@dataclass
class EntityMemory:
    value: str
    entity_type: EntityType
    first_mentioned: float
    last_mentioned: float
    mention_count: int = 1
    context: List[str] = None
    relationships: Dict[str, List[str]] = None
    importance: float = 0.5
    
    def __post_init__(self):
        if self.context is None:
            self.context = []
        if self.relationships is None:
            self.relationships = {}

class ConversationMemory:
    def __init__(self, 
                 max_short_term_turns=20, 
                 max_long_term_items=1000,
                 importance_threshold=0.7,
                 decay_factor=0.95,
                 consolidation_interval=3600):  # 1 hour
        
        # Core memory storage
        self.short_term_history = deque(maxlen=max_short_term_turns)
        self.long_term_memory = []
        self.episodic_memory = []  # Significant events/conversations
        self.semantic_memory = {}  # Facts and knowledge
        self.procedural_memory = {}  # How-to knowledge and patterns
        
        # Entity and relationship tracking
        self.entities = {}
        self.entity_relationships = defaultdict(lambda: defaultdict(float))
        
        # Topic and context tracking
        self.current_topics = []
        self.topic_history = deque(maxlen=50)
        self.conversation_context = {}
        
        # User profiling
        self.user_preferences = {}
        self.user_goals = []
        self.user_problems = []
        self.personality_traits = {}
        
        # Memory management parameters
        self.max_long_term_items = max_long_term_items
        self.importance_threshold = importance_threshold
        self.decay_factor = decay_factor
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        
        # Statistics and analytics
        self.conversation_stats = {
            'total_turns': 0,
            'total_tokens': 0,
            'session_start': time.time(),
            'topics_discussed': set(),
            'entities_mentioned': set()
        }

    def add_turn(self, user_query: str, assistant_response: str, 
                 extracted_entities: Dict[str, List[str]] = None,
                 sentiment: str = "neutral", importance: float = 0.5,
                 topics: List[str] = None):
        """Add a conversation turn with enhanced metadata"""
        
        timestamp = time.time()
        turn_data = {
            "user": user_query,
            "assistant": assistant_response,
            "timestamp": timestamp,
            "turn_index": self.conversation_stats['total_turns'],
            "entities": extracted_entities or {},
            "sentiment": sentiment,
            "importance": importance,
            "topics": topics or [],
            "session_id": self._get_session_id()
        }
        
        # Add to short-term memory
        self.short_term_history.append(turn_data)
        
        # Process entities
        if extracted_entities:
            self._process_entities(extracted_entities, timestamp, user_query + " " + assistant_response)
        
        # Process topics
        if topics:
            self._process_topics(topics, timestamp, turn_data)
        
        # Extract and store preferences, goals, problems
        self._extract_user_insights(user_query, assistant_response, timestamp)
        
        # Update statistics
        self._update_statistics(turn_data)
        
        # Check if turn should be consolidated to long-term memory
        if importance >= self.importance_threshold or self._is_significant_turn(turn_data):
            self._add_to_long_term(turn_data)
        
        # Periodic memory consolidation
        if timestamp - self.last_consolidation > self.consolidation_interval:
            self._consolidate_memory()

    def _process_entities(self, entities: Dict[str, List[str]], timestamp: float, context: str):
        """Process and update entity memories"""
        for entity_type_str, entity_list in entities.items():
            try:
                entity_type = EntityType(entity_type_str.lower())
            except ValueError:
                entity_type = EntityType.CONCEPT  # Default fallback
                
            for entity_value in entity_list:
                if not entity_value or not entity_value.strip():
                    continue
                    
                entity_key = f"{entity_type.value}:{entity_value.lower()}"
                
                if entity_key in self.entities:
                    # Update existing entity
                    entity_mem = self.entities[entity_key]
                    entity_mem.last_mentioned = timestamp
                    entity_mem.mention_count += 1
                    entity_mem.context.append(context[:200])  # Limit context length
                    entity_mem.importance = min(1.0, entity_mem.importance + 0.1)
                else:
                    # Create new entity memory
                    self.entities[entity_key] = EntityMemory(
                        value=entity_value,
                        entity_type=entity_type,
                        first_mentioned=timestamp,
                        last_mentioned=timestamp,
                        context=[context[:200]],
                        importance=0.5
                    )
                
                # Update conversation stats
                self.conversation_stats['entities_mentioned'].add(entity_value)

    def _process_topics(self, topics: List[str], timestamp: float, turn_data: Dict):
        """Process and track conversation topics"""
        for topic in topics:
            topic_lower = topic.lower().strip()
            if not topic_lower:
                continue
                
            # Add to current topics if not already present
            if topic_lower not in [t['name'] for t in self.current_topics]:
                self.current_topics.append({
                    'name': topic_lower,
                    'started': timestamp,
                    'last_mentioned': timestamp,
                    'mention_count': 1,
                    'importance': 0.5
                })
            else:
                # Update existing topic
                for t in self.current_topics:
                    if t['name'] == topic_lower:
                        t['last_mentioned'] = timestamp
                        t['mention_count'] += 1
                        t['importance'] = min(1.0, t['importance'] + 0.1)
                        break
            
            # Add to topic history
            self.topic_history.append({
                'topic': topic_lower,
                'timestamp': timestamp,
                'turn_index': turn_data['turn_index']
            })
            
            # Update conversation stats
            self.conversation_stats['topics_discussed'].add(topic_lower)

    def _extract_user_insights(self, user_query: str, assistant_response: str, timestamp: float):
        """Extract user preferences, goals, and problems from conversation"""
        user_lower = user_query.lower()
        
        # Simple keyword-based extraction (can be enhanced with NLP)
        preference_indicators = ['i like', 'i prefer', 'i love', 'i enjoy', 'i hate', 'i dislike']
        goal_indicators = ['i want to', 'i need to', 'my goal is', 'i hope to', 'i plan to']
        problem_indicators = ['i have trouble', 'i struggle with', 'problem with', 'issue with', 'difficulty']
        
        for indicator in preference_indicators:
            if indicator in user_lower:
                preference = user_query[user_lower.find(indicator) + len(indicator):].split('.')[0].strip()
                if preference:
                    self.user_preferences[preference] = {
                        'timestamp': timestamp,
                        'positive': indicator not in ['i hate', 'i dislike'],
                        'context': user_query[:100]
                    }
        
        for indicator in goal_indicators:
            if indicator in user_lower:
                goal = user_query[user_lower.find(indicator) + len(indicator):].split('.')[0].strip()
                if goal and len(goal) > 3:
                    self.user_goals.append({
                        'goal': goal,
                        'timestamp': timestamp,
                        'status': 'active',
                        'context': user_query[:100]
                    })
        
        for indicator in problem_indicators:
            if indicator in user_lower:
                problem = user_query[user_lower.find(indicator) + len(indicator):].split('.')[0].strip()
                if problem and len(problem) > 3:
                    self.user_problems.append({
                        'problem': problem,
                        'timestamp': timestamp,
                        'status': 'unresolved',
                        'context': user_query[:100]
                    })

    def _is_significant_turn(self, turn_data: Dict) -> bool:
        """Determine if a turn is significant enough for long-term storage"""
        # Check various significance criteria
        criteria_met = 0
        
        # High sentiment intensity
        if turn_data.get('sentiment') in ['very_positive', 'very_negative']:
            criteria_met += 1
        
        # Many entities mentioned
        if len(turn_data.get('entities', {})) > 2:
            criteria_met += 1
        
        # Long conversation turn
        if len(turn_data.get('user', '')) + len(turn_data.get('assistant', '')) > 500:
            criteria_met += 1
        
        # Important topics
        important_topics = ['goal', 'problem', 'plan', 'decision', 'important', 'urgent']
        if any(topic in turn_data.get('user', '').lower() for topic in important_topics):
            criteria_met += 1
        
        return criteria_met >= 2

    def _add_to_long_term(self, turn_data: Dict):
        """Add significant turns to long-term memory"""
        memory_item = MemoryItem(
            content=json.dumps(turn_data),
            memory_type=MemoryType.EPISODIC,
            timestamp=turn_data['timestamp'],
            importance=turn_data.get('importance', 0.5),
            tags=turn_data.get('topics', []),
            metadata=turn_data
        )
        
        self.long_term_memory.append(memory_item)
        
        # Manage memory size
        if len(self.long_term_memory) > self.max_long_term_items:
            # Remove least important items
            self.long_term_memory.sort(key=lambda x: x.importance * (1 - self._get_decay_factor(x.timestamp)))
            self.long_term_memory = self.long_term_memory[int(self.max_long_term_items * 0.1):]

    def _consolidate_memory(self):
        """Consolidate and organize memories"""
        current_time = time.time()
        
        # Apply decay to importance scores
        for memory in self.long_term_memory:
            age_factor = self._get_decay_factor(memory.timestamp)
            memory.importance *= age_factor
        
        # Clean up old, unimportant memories
        self.long_term_memory = [m for m in self.long_term_memory if m.importance > 0.1]
        
        # Update entity importance based on recency and frequency
        for entity in self.entities.values():
            age_factor = self._get_decay_factor(entity.last_mentioned)
            frequency_factor = min(1.0, entity.mention_count / 10.0)
            entity.importance = (entity.importance * age_factor + frequency_factor) / 2
        
        # Clean up inactive topics from current topics
        active_topics = []
        for topic in self.current_topics:
            if current_time - topic['last_mentioned'] < 1800:  # 30 minutes
                active_topics.append(topic)
        self.current_topics = active_topics
        
        self.last_consolidation = current_time

    def _get_decay_factor(self, timestamp: float) -> float:
        """Calculate decay factor based on age"""
        age_hours = (time.time() - timestamp) / 3600
        return self.decay_factor ** age_hours

    def _update_statistics(self, turn_data: Dict):
        """Update conversation statistics"""
        self.conversation_stats['total_turns'] += 1
        self.conversation_stats['total_tokens'] += len(turn_data.get('user', '').split()) + len(turn_data.get('assistant', '').split())

    def _get_session_id(self) -> str:
        """Generate a session ID based on start time"""
        return hashlib.md5(str(self.conversation_stats['session_start']).encode()).hexdigest()[:8]

    def get_relevant_context(self, query: str, max_items: int = 10) -> Dict[str, Any]:
        """Get relevant context for a query"""
        query_lower = query.lower()
        relevant_context = {
            'recent_history': list(self.short_term_history)[-5:],
            'relevant_entities': [],
            'relevant_memories': [],
            'current_topics': self.current_topics,
            'user_profile': self._get_user_profile(),
            'conversation_summary': self._get_conversation_summary()
        }
        
        # Find relevant entities
        for entity_key, entity in self.entities.items():
            if any(word in query_lower for word in entity.value.lower().split()):
                relevant_context['relevant_entities'].append(entity)
        
        # Find relevant long-term memories
        for memory in self.long_term_memory:
            if any(tag in query_lower for tag in memory.tags):
                memory.access_count += 1
                memory.last_accessed = time.time()
                relevant_context['relevant_memories'].append(memory)
        
        # Sort by relevance and limit
        relevant_context['relevant_entities'] = sorted(
            relevant_context['relevant_entities'], 
            key=lambda x: x.importance, 
            reverse=True
        )[:max_items//2]
        
        relevant_context['relevant_memories'] = sorted(
            relevant_context['relevant_memories'], 
            key=lambda x: x.importance * (2 if x.access_count > 1 else 1), 
            reverse=True
        )[:max_items//2]
        
        return relevant_context

    def _get_user_profile(self) -> Dict[str, Any]:
        """Generate user profile from accumulated data"""
        return {
            'preferences': dict(list(self.user_preferences.items())[-10:]),  # Last 10 preferences
            'goals': [g for g in self.user_goals if g['status'] == 'active'][-5:],
            'problems': [p for p in self.user_problems if p['status'] == 'unresolved'][-5:],
            'personality_traits': self.personality_traits,
            'frequent_topics': self._get_frequent_topics(5),
            'conversation_style': self._analyze_conversation_style()
        }

    def _get_frequent_topics(self, limit: int = 5) -> List[str]:
        """Get most frequently discussed topics"""
        topic_counts = defaultdict(int)
        for topic_entry in self.topic_history:
            topic_counts[topic_entry['topic']] += 1
        
        return sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def _analyze_conversation_style(self) -> Dict[str, Any]:
        """Analyze user's conversation style"""
        if not self.short_term_history:
            return {}
        
        total_user_length = sum(len(turn.get('user', '')) for turn in self.short_term_history)
        avg_user_length = total_user_length / len(self.short_term_history)
        
        question_count = sum(1 for turn in self.short_term_history if '?' in turn.get('user', ''))
        question_ratio = question_count / len(self.short_term_history)
        
        return {
            'avg_message_length': avg_user_length,
            'question_ratio': question_ratio,
            'engagement_level': 'high' if avg_user_length > 50 else 'medium' if avg_user_length > 20 else 'low'
        }

    def _get_conversation_summary(self) -> str:
        """Generate a brief conversation summary"""
        if not self.short_term_history:
            return "No conversation history available."
        
        recent_topics = [t['name'] for t in self.current_topics[:3]]
        entity_count = len(self.entities)
        turn_count = len(self.short_term_history)
        
        summary = f"Conversation with {turn_count} recent turns"
        if recent_topics:
            summary += f", discussing: {', '.join(recent_topics)}"
        if entity_count:
            summary += f". {entity_count} entities tracked"
        
        return summary

    def search_memory(self, query: str, memory_types: List[MemoryType] = None, limit: int = 10) -> List[Any]:
        """Search through all memory types"""
        results = []
        query_lower = query.lower()
        
        # Search short-term history
        for turn in self.short_term_history:
            if query_lower in turn.get('user', '').lower() or query_lower in turn.get('assistant', '').lower():
                results.append(('short_term', turn))
        
        # Search long-term memory
        for memory in self.long_term_memory:
            if not memory_types or memory.memory_type in memory_types:
                if query_lower in memory.content.lower() or any(query_lower in tag for tag in memory.tags):
                    results.append(('long_term', memory))
        
        # Search entities
        for entity_key, entity in self.entities.items():
            if query_lower in entity.value.lower() or any(query_lower in ctx.lower() for ctx in entity.context):
                results.append(('entity', entity))
        
        return results[:limit]

    def export_memory(self, include_statistics: bool = True) -> Dict[str, Any]:
        """Export memory data for persistence"""
        export_data = {
            'short_term_history': list(self.short_term_history),
            'long_term_memory': [asdict(m) for m in self.long_term_memory],
            'entities': {k: asdict(v) for k, v in self.entities.items()},
            'current_topics': self.current_topics,
            'topic_history': list(self.topic_history),
            'user_preferences': self.user_preferences,
            'user_goals': self.user_goals,
            'user_problems': self.user_problems,
            'personality_traits': self.personality_traits,
            'conversation_context': self.conversation_context,
            'export_timestamp': time.time()
        }
        
        if include_statistics:
            export_data['statistics'] = dict(self.conversation_stats)
            export_data['statistics']['topics_discussed'] = list(self.conversation_stats['topics_discussed'])
            export_data['statistics']['entities_mentioned'] = list(self.conversation_stats['entities_mentioned'])
        
        return export_data

    def import_memory(self, memory_data: Dict[str, Any]):
        """Import memory data from persistence"""
        if 'short_term_history' in memory_data:
            self.short_term_history = deque(memory_data['short_term_history'], maxlen=self.short_term_history.maxlen)
        
        if 'long_term_memory' in memory_data:
            self.long_term_memory = [MemoryItem(**item) for item in memory_data['long_term_memory']]
        
        if 'entities' in memory_data:
            self.entities = {k: EntityMemory(**v) for k, v in memory_data['entities'].items()}
        
        # Import other components
        for key in ['current_topics', 'topic_history', 'user_preferences', 'user_goals', 
                   'user_problems', 'personality_traits', 'conversation_context']:
            if key in memory_data:
                setattr(self, key, memory_data[key])
        
        if 'statistics' in memory_data:
            stats = memory_data['statistics']
            self.conversation_stats.update(stats)
            if 'topics_discussed' in stats:
                self.conversation_stats['topics_discussed'] = set(stats['topics_discussed'])
            if 'entities_mentioned' in stats:
                self.conversation_stats['entities_mentioned'] = set(stats['entities_mentioned'])

class PersistentMemoryStore:
    """Enhanced memory store with persistence capabilities"""
    
    def __init__(self, storage_path: str = "memory_store"):
        self.storage_path = storage_path
        self.sessions = {}
        self.lock = threading.Lock()
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Ensure storage directory exists"""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    
    def _get_user_file_path(self, user_id: str) -> str:
        """Get file path for user memory"""
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('_', '-'))
        return os.path.join(self.storage_path, f"{safe_user_id}_memory.json")
    
    def get(self, user_id: str) -> ConversationMemory:
        """Get or create user memory"""
        with self.lock:
            if user_id not in self.sessions:
                memory = ConversationMemory()
                
                # Try to load from disk
                file_path = self._get_user_file_path(user_id)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            memory_data = json.load(f)
                        memory.import_memory(memory_data)
                    except Exception as e:
                        print(f"Error loading memory for user {user_id}: {e}")
                
                self.sessions[user_id] = memory
            
            return self.sessions[user_id]
    
    def save(self, user_id: str, memory: ConversationMemory):
        """Save user memory to disk and cache"""
        with self.lock:
            self.sessions[user_id] = memory
            
            # Save to disk
            file_path = self._get_user_file_path(user_id)
            try:
                memory_data = memory.export_memory()
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving memory for user {user_id}: {e}")
    
    def clear(self, user_id: str):
        """Clear user memory from cache and disk"""
        with self.lock:
            self.sessions.pop(user_id, None)
            file_path = self._get_user_file_path(user_id)
            if os.path.exists(file_path):
                os.remove(file_path)
    
    def all_user_ids(self) -> List[str]:
        """Get all user IDs with stored memories"""
        user_ids = set(self.sessions.keys())
        
        # Add user IDs from disk
        if os.path.exists(self.storage_path):
            for filename in os.listdir(self.storage_path):
                if filename.endswith('_memory.json'):
                    user_id = filename[:-12]  # Remove '_memory.json'
                    user_ids.add(user_id)
        
        return list(user_ids)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get overall memory statistics"""
        stats = {
            'total_users': len(self.all_user_ids()),
            'active_sessions': len(self.sessions),
            'storage_path': self.storage_path,
            'user_stats': {}
        }
        
        for user_id in self.all_user_ids():
            memory = self.get(user_id)
            stats['user_stats'][user_id] = {
                'total_turns': memory.conversation_stats['total_turns'],
                'entities_count': len(memory.entities),
                'long_term_memories': len(memory.long_term_memory),
                'session_start': memory.conversation_stats['session_start']
            }
        
        return stats

class InMemoryUserMemoryStore:
    """Original in-memory store for backward compatibility"""
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def get(self, user_id):
        with self.lock:
            if user_id not in self.sessions:
                self.sessions[user_id] = ConversationMemory()
            return self.sessions[user_id]

    def save(self, user_id, memory: ConversationMemory):
        with self.lock:
            self.sessions[user_id] = memory

    def clear(self, user_id):
        with self.lock:
            self.sessions.pop(user_id, None)

    def all_user_ids(self):
        return list(self.sessions.keys())
