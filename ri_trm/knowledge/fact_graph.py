"""
Layer 1: Factual Knowledge Graph (K_F)

External world knowledge for factual grounding during reasoning.
Stores entities, relations, and facts that can be queried during generation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import json
import sqlite3
from abc import ABC, abstractmethod


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    id: str
    name: str
    type: str
    properties: Dict[str, Any]


@dataclass
class Relation:
    """A relation between entities"""
    id: str
    name: str
    source: str  # entity id
    target: str  # entity id
    properties: Dict[str, Any]


@dataclass
class Fact:
    """A factual statement (entity, relation, entity)"""
    subject: str  # entity id
    predicate: str  # relation name
    object: str  # entity id
    confidence: float = 1.0
    source: Optional[str] = None


class FactualQueryEngine(ABC):
    """Abstract interface for querying factual knowledge"""
    
    @abstractmethod
    def query_facts(self, subject: str, predicate: str = None, object: str = None) -> List[Fact]:
        """Query facts matching the pattern"""
        pass
    
    @abstractmethod
    def get_entity_info(self, entity_id: str) -> Optional[Entity]:
        """Get information about an entity"""
        pass
    
    @abstractmethod
    def find_related_entities(self, entity_id: str, relation_type: str = None) -> List[Entity]:
        """Find entities related to the given entity"""
        pass


class SQLiteFactStore(FactualQueryEngine):
    """SQLite-based storage for factual knowledge"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for entities, relations, and facts"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT  -- JSON
            );
            
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                properties TEXT,  -- JSON
                FOREIGN KEY (source) REFERENCES entities (id),
                FOREIGN KEY (target) REFERENCES entities (id)
            );
            
            CREATE TABLE IF NOT EXISTS facts (
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT,
                PRIMARY KEY (subject, predicate, object),
                FOREIGN KEY (subject) REFERENCES entities (id),
                FOREIGN KEY (object) REFERENCES entities (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts (subject);
            CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts (predicate);
            CREATE INDEX IF NOT EXISTS idx_facts_object ON facts (object);
        """)
        self.conn.commit()
    
    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge base"""
        self.conn.execute(
            "INSERT OR REPLACE INTO entities (id, name, type, properties) VALUES (?, ?, ?, ?)",
            (entity.id, entity.name, entity.type, json.dumps(entity.properties))
        )
        self.conn.commit()
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge base"""
        self.conn.execute(
            "INSERT OR REPLACE INTO facts (subject, predicate, object, confidence, source) VALUES (?, ?, ?, ?, ?)",
            (fact.subject, fact.predicate, fact.object, fact.confidence, fact.source)
        )
        self.conn.commit()
    
    def query_facts(self, subject: str = None, predicate: str = None, object: str = None) -> List[Fact]:
        """Query facts matching the pattern"""
        conditions = []
        params = []
        
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object:
            conditions.append("object = ?")
            params.append(object)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM facts WHERE {where_clause}"
        
        cursor = self.conn.execute(query, params)
        facts = []
        
        for row in cursor:
            facts.append(Fact(
                subject=row['subject'],
                predicate=row['predicate'],
                object=row['object'],
                confidence=row['confidence'],
                source=row['source']
            ))
        
        return facts
    
    def get_entity_info(self, entity_id: str) -> Optional[Entity]:
        """Get information about an entity"""
        cursor = self.conn.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()
        
        if row:
            return Entity(
                id=row['id'],
                name=row['name'],
                type=row['type'],
                properties=json.loads(row['properties']) if row['properties'] else {}
            )
        return None
    
    def find_related_entities(self, entity_id: str, relation_type: str = None) -> List[Entity]:
        """Find entities related to the given entity"""
        if relation_type:
            query = """
                SELECT DISTINCT e.* FROM entities e
                JOIN facts f ON (e.id = f.object OR e.id = f.subject)
                WHERE (f.subject = ? OR f.object = ?) AND f.predicate = ? AND e.id != ?
            """
            params = (entity_id, entity_id, relation_type, entity_id)
        else:
            query = """
                SELECT DISTINCT e.* FROM entities e
                JOIN facts f ON (e.id = f.object OR e.id = f.subject)
                WHERE (f.subject = ? OR f.object = ?) AND e.id != ?
            """
            params = (entity_id, entity_id, entity_id)
        
        cursor = self.conn.execute(query, params)
        entities = []
        
        for row in cursor:
            entities.append(Entity(
                id=row['id'],
                name=row['name'],
                type=row['type'],
                properties=json.loads(row['properties']) if row['properties'] else {}
            ))
        
        return entities


class FactualKnowledgeGraph(nn.Module):
    """
    Layer 1: Factual Knowledge Graph (K_F)
    
    Stores external world knowledge as (entity, relation, entity) triples.
    Used for factual grounding during reasoning when domain facts are needed.
    
    Examples for Code Generation:
    ("numpy.array", "returns", "ndarray")
    ("pandas.DataFrame", "requires", "pandas>=1.0")
    ("async def", "requires", "Python>=3.5")
    """
    
    def __init__(
        self,
        domain: str,
        embedding_dim: int = 512,
        db_path: str = ":memory:",
        max_cached_entities: int = 1000
    ):
        super().__init__()
        self.domain = domain
        self.embedding_dim = embedding_dim
        
        # Factual storage
        self.fact_store = SQLiteFactStore(db_path)
        
        # Neural components for embedding facts
        self.entity_embedding = nn.Embedding(max_cached_entities, embedding_dim)
        self.relation_embedding = nn.Embedding(100, embedding_dim)  # Common relations
        
        # Caches for embedding lookup
        self.entity_to_id: Dict[str, int] = {}
        self.relation_to_id: Dict[str, int] = {}
        self.next_entity_id = 0
        self.next_relation_id = 0
        
        # Pre-populate common relations
        self._init_common_relations()
    
    def _init_common_relations(self):
        """Initialize common relation types"""
        common_relations = [
            "returns", "requires", "is_a", "has_property", "used_by",
            "depends_on", "implements", "extends", "contains", "produces"
        ]
        
        for relation in common_relations:
            self._get_relation_id(relation)
    
    def _get_entity_id(self, entity: str) -> int:
        """Get or create embedding ID for entity"""
        if entity not in self.entity_to_id:
            self.entity_to_id[entity] = self.next_entity_id % self.entity_embedding.num_embeddings
            self.next_entity_id += 1
        return self.entity_to_id[entity]
    
    def _get_relation_id(self, relation: str) -> int:
        """Get or create embedding ID for relation"""
        if relation not in self.relation_to_id:
            self.relation_to_id[relation] = self.next_relation_id % self.relation_embedding.num_embeddings
            self.next_relation_id += 1
        return self.relation_to_id[relation]
    
    def add_entity(self, entity: Entity):
        """Add an entity to the knowledge graph"""
        self.fact_store.add_entity(entity)
        # Pre-cache entity embedding
        self._get_entity_id(entity.id)
    
    def add_fact(self, fact: Fact):
        """Add a fact to the knowledge graph"""
        self.fact_store.add_fact(fact)
        # Pre-cache embeddings
        self._get_entity_id(fact.subject)
        self._get_entity_id(fact.object)
        self._get_relation_id(fact.predicate)
    
    def add_facts_batch(self, facts: List[Fact]):
        """Add multiple facts efficiently"""
        for fact in facts:
            self.add_fact(fact)
    
    def query_relevant_facts(
        self,
        context_entities: List[str],
        max_facts: int = 10
    ) -> List[Fact]:
        """
        Query facts relevant to given context entities
        
        Args:
            context_entities: List of entity names mentioned in context
            max_facts: Maximum number of facts to return
            
        Returns:
            List of relevant facts
        """
        all_facts = []
        
        for entity in context_entities:
            # Find facts where entity is subject or object
            subject_facts = self.fact_store.query_facts(subject=entity)
            object_facts = self.fact_store.query_facts(object=entity)
            all_facts.extend(subject_facts + object_facts)
        
        # Remove duplicates and sort by confidence
        unique_facts = {(f.subject, f.predicate, f.object): f for f in all_facts}
        sorted_facts = sorted(unique_facts.values(), key=lambda f: f.confidence, reverse=True)
        
        return sorted_facts[:max_facts]
    
    def embed_facts(self, facts: List[Fact]) -> torch.Tensor:
        """
        Convert facts to neural embeddings
        
        Args:
            facts: List of facts to embed
            
        Returns:
            Fact embeddings [F, 3*D] where F is number of facts
        """
        if not facts:
            return torch.empty(0, 3 * self.embedding_dim)
        
        embeddings = []
        
        for fact in facts:
            # Get embeddings for subject, predicate, object
            subj_id = self._get_entity_id(fact.subject)
            pred_id = self._get_relation_id(fact.predicate)
            obj_id = self._get_entity_id(fact.object)
            
            subj_emb = self.entity_embedding(torch.tensor(subj_id))
            pred_emb = self.relation_embedding(torch.tensor(pred_id))
            obj_emb = self.entity_embedding(torch.tensor(obj_id))
            
            # Concatenate (subject, predicate, object) embeddings
            fact_emb = torch.cat([subj_emb, pred_emb, obj_emb], dim=0)
            embeddings.append(fact_emb)
        
        return torch.stack(embeddings)
    
    def query_and_embed(
        self,
        context_entities: List[str],
        max_facts: int = 10
    ) -> Optional[torch.Tensor]:
        """
        Query relevant facts and return their embeddings
        
        Args:
            context_entities: Entities mentioned in current context
            max_facts: Maximum facts to retrieve
            
        Returns:
            Embedded facts [F, 3*D] or None if no facts found
        """
        facts = self.query_relevant_facts(context_entities, max_facts)
        
        if not facts:
            return None
        
        return self.embed_facts(facts)
    
    def get_entity_context(self, entity: str) -> Dict[str, Any]:
        """
        Get contextual information about an entity
        
        Args:
            entity: Entity identifier
            
        Returns:
            Dictionary with entity info and related facts
        """
        entity_info = self.fact_store.get_entity_info(entity)
        related_facts = self.fact_store.query_facts(subject=entity)
        related_facts.extend(self.fact_store.query_facts(object=entity))
        
        return {
            "entity": entity_info,
            "facts": related_facts,
            "related_entities": self.fact_store.find_related_entities(entity)
        }
    
    def load_domain_knowledge(self, knowledge_file: str):
        """
        Load domain-specific knowledge from file
        
        Args:
            knowledge_file: JSON file with entities and facts
        """
        try:
            with open(knowledge_file, 'r') as f:
                data = json.load(f)
            
            # Load entities
            for entity_data in data.get('entities', []):
                entity = Entity(**entity_data)
                self.add_entity(entity)
            
            # Load facts
            for fact_data in data.get('facts', []):
                fact = Fact(**fact_data)
                self.add_fact(fact)
                
        except Exception as e:
            print(f"Warning: Could not load knowledge from {knowledge_file}: {e}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        entity_count = self.fact_store.conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        fact_count = self.fact_store.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        
        return {
            "domain": self.domain,
            "entity_count": entity_count,
            "fact_count": fact_count,
            "cached_entities": len(self.entity_to_id),
            "cached_relations": len(self.relation_to_id),
            "embedding_usage": {
                "entities": len(self.entity_to_id) / self.entity_embedding.num_embeddings,
                "relations": len(self.relation_to_id) / self.relation_embedding.num_embeddings
            }
        }
    
    def forward(self, context_entities: List[str]) -> Optional[torch.Tensor]:
        """
        Forward pass: query and embed relevant facts
        
        Args:
            context_entities: Entities in current context
            
        Returns:
            Embedded relevant facts or None
        """
        return self.query_and_embed(context_entities)