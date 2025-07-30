from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import numpy as np
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import pandas as pd
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BlipProcessor, BlipForConditionalGeneration
import torch
import re
import warnings
from typing import Dict, List, Tuple, Optional
import time
import requests
from PIL import Image
import io
import threading
import base64

warnings.filterwarnings('ignore')

# Flask app configuration
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS = set(os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif").split(","))
PORT = int(os.environ.get("PORT", "10000"))

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Neo4j Configuration
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
MODEL_CHOICE = os.environ.get("MODEL_CHOICE", "phi3.5")

# Plant Knowledge Graph Class with Enhanced Capabilities
class EnhancedPlantKnowledgeGraph:
    def __init__(self, uri: str, username: str, password: str, model_choice: str = "phi3.5"):
    self.uri = uri
    self.username = username
    self.password = password
    self.driver = None
    self.model_choice = model_choice
    self.connection_tested = False
    self.data_loaded = False
    self.last_error = None
    
    # Initialize models as None - will be loaded in background
    self.image_model = None
    self.image_processor = None
    self.text_generator = None
    self.models_loading = False
    self.models_loaded = False
    
    # Start background model loading
    self.start_background_model_loading()
    
    # Try to establish connection with multiple URI formats
    self._establish_connection()
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.model_choice = model_choice
        self.connection_tested = False
        self.data_loaded = False
        self.last_error = None
        
        # Initialize models for plant identification and data generation
        self.image_model = None
        self.image_processor = None
        self.text_generator = None
        
        # Try to establish connection with multiple URI formats
        self._establish_connection()
    



    def start_background_model_loading(self):
        """Start loading AI models in background thread"""
        self.models_loading = True
        thread = threading.Thread(target=self.load_ai_models_background, daemon=True)
        thread.start()

    def load_ai_models_background(self):
        """Load AI models in background thread"""
        try:
            print("🤖 Loading AI models in background...")
            
            # Load BLIP model for image captioning/plant identification
            self.image_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.image_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # Load lightweight text generation model
            self.text_generator = pipeline(
                "text-generation", 
                model="microsoft/DialoGPT-medium",
                tokenizer="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models_loaded = True
            self.models_loading = False
            print("✅ AI models loaded successfully in background")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load AI models in background: {e}")
            print("🔄 Falling back to API-based solutions")
            self.models_loading = False
            self.models_loaded = False
    
    def identify_plant_from_image(self, image_path: str) -> Tuple[str, float]:
        """Identify plant from image using BLIP model or fallback methods"""
        try:
            # Check if models are still loading
            if self.models_loading:
                return "Models still loading, please try again in a moment", 0.1
            
            if self.models_loaded and self.image_model and self.image_processor:
                # Use local BLIP model
                image = Image.open(image_path).convert('RGB')
                
                # Generate caption with plant focus
                inputs = self.image_processor(image, "a photo of", return_tensors="pt")
                out = self.image_model.generate(**inputs, max_length=50)
                caption = self.image_processor.decode(out[0], skip_special_tokens=True)
                
                # Extract potential plant name from caption
                plant_keywords = ['plant', 'flower', 'leaf', 'tree', 'herb', 'grass', 'fern', 'moss']
                confidence = 0.7
                
                # Simple plant name extraction logic
                words = caption.lower().split()
                for i, word in enumerate(words):
                    if word in plant_keywords and i < len(words) - 1:
                        potential_name = words[i + 1]
                        return potential_name.capitalize(), confidence
                
                return caption.replace("a photo of", "").strip(), 0.5
            
            else:
                # Fallback: Return generic plant identification
                return "Plant species identification unavailable", 0.2
                    
        except Exception as e:
            print(f"❌ Plant identification failed: {e}")
            return "Unknown Plant", 0.1
    
    def generate_plant_data_from_web(self, plant_name: str) -> Dict:
        """Generate comprehensive plant data using web search and LLM"""
        try:
            # First, try to get basic information from free botanical APIs
            plant_data = self.fetch_plant_data_from_apis(plant_name)
            
            # If no data found, generate using LLM
            if not plant_data or plant_data.get('scientific_name') == 'Unknown':
                plant_data = self.generate_plant_data_with_llm(plant_name)
            
            return plant_data
            
        except Exception as e:
            print(f"❌ Data generation failed for {plant_name}: {e}")
            return self.create_minimal_plant_data(plant_name)
    
    def fetch_plant_data_from_apis(self, plant_name: str) -> Dict:
        """Fetch plant data from free botanical APIs"""
        try:
            # Try GBIF API (Global Biodiversity Information Facility)
            gbif_url = f"https://api.gbif.org/v1/species/search?q={plant_name}&limit=1"
            response = requests.get(gbif_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    
                    plant_data = {
                        'plant_name': plant_name,
                        'scientific_name': result.get('scientificName', 'Unknown'),
                        'family': result.get('family', 'Unknown Family'),
                        'kingdom': result.get('kingdom', 'Plantae'),
                        'order': result.get('order', 'Unknown Order'),
                        'genus': result.get('genus', 'Unknown Genus'),
                        'species': result.get('species', 'Unknown Species'),
                        'medicinal_properties': '',
                        'habitat': '',
                        'uses': '',
                        'chemical_components': ''
                    }
                    
                    # Generate additional fields using LLM
                    additional_data = self.generate_additional_plant_info(plant_name, plant_data['scientific_name'])
                    plant_data.update(additional_data)
                    
                    print(f"✅ Fetched data for {plant_name} from GBIF API")
                    return plant_data
            
            # Fallback to other APIs like Trefle, iNaturalist, etc.
            return self.try_alternative_apis(plant_name)
            
        except Exception as e:
            print(f"⚠️ API fetch failed for {plant_name}: {e}")
            return {}
    
    def try_alternative_apis(self, plant_name: str) -> Dict:
        """Try alternative free botanical APIs"""
        try:
            # Try Wikipedia API for basic information
            wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{plant_name}"
            response = requests.get(wiki_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                extract = data.get('extract', '')
                
                # Extract basic information from Wikipedia summary
                plant_data = {
                    'plant_name': plant_name,
                    'scientific_name': self.extract_scientific_name(extract),
                    'family': 'Unknown Family',
                    'kingdom': 'Plantae',
                    'order': 'Unknown Order',
                    'genus': plant_name.split()[0] if ' ' in plant_name else 'Unknown Genus',
                    'species': plant_name.split()[1] if len(plant_name.split()) > 1 else 'Unknown Species',
                    'medicinal_properties': '',
                    'habitat': extract[:200] if extract else '',
                    'uses': '',
                    'chemical_components': ''
                }
                
                # Generate missing fields
                additional_data = self.generate_additional_plant_info(plant_name, plant_data['scientific_name'])
                plant_data.update(additional_data)
                
                return plant_data
                
        except Exception as e:
            print(f"⚠️ Alternative API failed: {e}")
        
        return {}
    
    def extract_scientific_name(self, text: str) -> str:
        """Extract scientific name from text using regex"""
        # Look for italicized text or text in parentheses that looks like scientific names
        scientific_pattern = r'\b[A-Z][a-z]+ [a-z]+\b'
        matches = re.findall(scientific_pattern, text)
        
        if matches:
            return matches[0]
        
        return 'Unknown'
    
    def generate_additional_plant_info(self, plant_name: str, scientific_name: str) -> Dict:
    """Generate additional plant information using LLM"""
    try:
        prompts = {
            'medicinal_properties': f"List the medicinal properties and health benefits of {plant_name} ({scientific_name}). Be concise.",
            'uses': f"What are the traditional, cultural, and industrial uses of {plant_name}? Provide a brief overview.",
            'chemical_components': f"What are the main chemical compounds found in {plant_name}? List key components."
        }
        
        additional_data = {}
        
        for field, prompt in prompts.items():
            try:
                # Check if models are loaded and available
                if self.models_loaded and self.text_generator:
                    # Use local model
                    response = self.text_generator(prompt, max_length=100, do_sample=True, temperature=0.7)
                    generated_text = response[0]['generated_text'].replace(prompt, '').strip()
                    additional_data[field] = self.clean_text(generated_text)
                elif self.models_loading:
                    # Models still loading, use template
                    additional_data[field] = f"AI models loading... Using template for {plant_name}"
                else:
                    # Fallback to predefined templates
                    additional_data[field] = self.get_template_info(plant_name, field)
                    
            except Exception as e:
                print(f"⚠️ LLM generation failed for {field}: {e}")
                additional_data[field] = self.get_template_info(plant_name, field)
        
        return additional_data
        
    except Exception as e:
        print(f"❌ Additional info generation failed: {e}")
        return {
            'medicinal_properties': 'Properties under research',
            'uses': 'Traditional and ornamental uses',
            'chemical_components': 'Various organic compounds'
        }
    
    def get_template_info(self, plant_name: str, field: str) -> str:
        """Provide template information when LLM is not available"""
        templates = {
            'medicinal_properties': f"Traditional medicinal uses of {plant_name} are being researched. May have anti-inflammatory and antioxidant properties.",
            'uses': f"{plant_name} is used for ornamental purposes, traditional medicine, and may have industrial applications.",
            'chemical_components': f"{plant_name} contains various phytochemicals including flavonoids, alkaloids, and essential oils."
        }
        
        return templates.get(field, 'Information being researched')
    
    def generate_plant_data_with_llm(self, plant_name: str) -> Dict:
        """Generate complete plant data using LLM when APIs fail"""
        try:
            base_prompt = f"""Generate detailed botanical information for the plant "{plant_name}". Include:
            - Scientific name
            - Family
            - Medicinal properties
            - Traditional uses
            - Habitat
            - Chemical components
            Format as structured data."""
            
            if self.text_generator:
                response = self.text_generator(base_prompt, max_length=200, do_sample=True, temperature=0.7)
                generated_text = response[0]['generated_text']
                
                # Parse the generated text to extract structured data
                return self.parse_llm_response(plant_name, generated_text)
            else:
                return self.create_minimal_plant_data(plant_name)
                
        except Exception as e:
            print(f"❌ LLM generation failed: {e}")
            return self.create_minimal_plant_data(plant_name)
    
    def parse_llm_response(self, plant_name: str, llm_text: str) -> Dict:
        """Parse LLM response to extract structured plant data"""
        # Simple parsing logic - in production, use more sophisticated NLP
        
        plant_data = {
            'plant_name': plant_name,
            'scientific_name': 'Unknown',
            'family': 'Unknown Family',
            'kingdom': 'Plantae',
            'order': 'Unknown Order',
            'genus': plant_name.split()[0] if ' ' in plant_name else 'Unknown',
            'species': plant_name.split()[1] if len(plant_name.split()) > 1 else 'Unknown',
            'medicinal_properties': 'Under research',
            'habitat': 'Various environments',
            'uses': 'Traditional and ornamental uses',
            'chemical_components': 'Various organic compounds'
        }
        
        # Extract scientific name if present
        scientific_match = re.search(r'[A-Z][a-z]+ [a-z]+', llm_text)
        if scientific_match:
            plant_data['scientific_name'] = scientific_match.group()
        
        return plant_data
    
    def create_minimal_plant_data(self, plant_name: str) -> Dict:
        """Create minimal plant data when all other methods fail"""
        return {
            'plant_name': plant_name,
            'scientific_name': f"{plant_name.replace(' ', '_')} sp.",
            'family': 'Unknown Family',
            'kingdom': 'Plantae',
            'order': 'Unknown Order',
            'genus': plant_name.split()[0] if ' ' in plant_name else plant_name,
            'species': plant_name.split()[1] if len(plant_name.split()) > 1 else 'sp.',
            'medicinal_properties': 'Medicinal properties under investigation',
            'habitat': 'Natural habitat varies by species',
            'uses': 'Traditional uses and ornamental purposes',
            'chemical_components': 'Contains various phytochemicals and organic compounds'
        }
    
    def search_or_generate_plant_data(self, plant_name: str) -> Tuple[bool, List[Dict], str]:
        """Search for plant data, generate if not found"""
        try:
            # First, try to search existing data
            success, results, response = self.search_plants(plant_name)
            
            if success and results:
                return success, results, response
            
            # If not found, generate new data
            print(f"🔄 Plant '{plant_name}' not found in KG. Generating new data...")
            
            # Generate comprehensive plant data
            new_plant_data = self.generate_plant_data_from_web(plant_name)
            
            # Insert into knowledge graph
            cypher_query = self.template_enhanced_cypher_insert(new_plant_data)
            insert_success, insert_message = self.insert_plant_data(cypher_query)
            
            if insert_success:
                print(f"✅ Successfully added {plant_name} to knowledge graph")
                
                # Now search again to return the newly added data
                success, results, response = self.search_plants(plant_name)
                
                if success and results:
                    response = f"🆕 Generated and added new plant data for '{plant_name}'\n\n" + response
                    return True, results, response
            
            # If insertion failed, return the generated data anyway
            results = [new_plant_data]
            response = self.format_search_results(results, plant_name)
            response = f"🆕 Generated plant data for '{plant_name}' (not saved to database)\n\n" + response
            
            return True, results, response
            
        except Exception as e:
            error_msg = f"Failed to search or generate data for '{plant_name}': {str(e)}"
            return False, [], error_msg

    # Include all previous methods from the original class
    def _establish_connection(self):
        """Try to establish connection with multiple URI formats"""
        uris_to_try = [self.uri]
        
        for uri in uris_to_try:
            try:
                print(f"🔄 Trying to connect to: {uri}")
                
                if 'neo4j+s' in uri or 'bolt+s' in uri:
                    configs_to_try = [{"uri": uri, "auth": (self.username, self.password)}]
                else:
                    configs_to_try = [
                        {"uri": uri, "auth": (self.username, self.password), "encrypted": True, "trust": "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"},
                        {"uri": uri, "auth": (self.username, self.password), "encrypted": True, "trust": "TRUST_ALL_CERTIFICATES"},
                        {"uri": uri, "auth": (self.username, self.password)}
                    ]
                
                for config in configs_to_try:
                    try:
                        config_name = config.get('trust', 'default')
                        print(f"  📋 Trying config: {config_name}")
                        
                        self.driver = GraphDatabase.driver(**config)
                        
                        with self.driver.session() as session:
                            result = session.run("RETURN 1 as test")
                            result.single()
                        
                        print(f"✅ Successfully connected to: {uri} with {config_name} config")
                        self.uri = uri
                        return
                        
                    except Exception as config_error:
                        print(f"  ❌ Config failed: {str(config_error)}")
                        self.last_error = str(config_error)
                        if self.driver:
                            try:
                                self.driver.close()
                            except:
                                pass
                            self.driver = None
                        continue
                
            except Exception as e:
                print(f"❌ URI {uri} failed completely: {str(e)}")
                self.last_error = str(e)
                continue
        
        print("❌ All connection attempts failed")
    
    def close(self):
        if hasattr(self, 'driver') and self.driver:
            try:
                self.driver.close()
            except:
                pass
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test if database connection is working"""
        if not self.driver:
            return False, f"No driver available. Last error: {self.last_error or 'Unknown connection error'}"
        
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection successful' as message, datetime() as timestamp")
                record = result.single()
                self.connection_tested = True
                return True, f"{record['message']} at {record['timestamp']}"
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            
            print("🔄 Connection failed, attempting to reconnect...")
            self._establish_connection()
            
            if self.driver:
                try:
                    with self.driver.session() as session:
                        result = session.run("RETURN 'Reconnected successfully' as message")
                        record = result.single()
                        self.connection_tested = True
                        return True, record['message']
                except Exception as e2:
                    return False, f"Reconnection also failed: {str(e2)}"
            
            return False, f"Database connection failed: {error_msg}"
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'["\\\n\r\t]', ' ', text)
        text = re.sub(r'\[[^\]]*\]', '', text)
        return text[:1000]
    
    def template_enhanced_cypher_insert(self, plant_data: Dict) -> str:
        """Template-based enhanced Cypher with relationships"""
        plant_name = self.clean_text(plant_data.get('plant_name', '')) or 'Unknown Plant'
        scientific_name = self.clean_text(plant_data.get('scientific_name', '')) or 'Unknown'
        family = self.clean_text(plant_data.get('family', '')) or 'Unknown Family'
        kingdom = self.clean_text(plant_data.get('kingdom', '')) or 'Plantae'
        order = self.clean_text(plant_data.get('order', '')) or 'Unknown Order'
        genus = self.clean_text(plant_data.get('genus', '')) or 'Unknown Genus'
        species = self.clean_text(plant_data.get('species', '')) or 'Unknown Species'
        medicinal_properties = self.clean_text(plant_data.get('medicinal_properties', ''))
        habitat = self.clean_text(plant_data.get('habitat', ''))
        uses = self.clean_text(plant_data.get('uses', ''))
        chemical_components = self.clean_text(plant_data.get('chemical_components', ''))
        
        cypher = f'''
        MERGE (k:Kingdom {{name: "{kingdom}"}})
        MERGE (f:Family {{name: "{family}"}})
        MERGE (g:Genus {{name: "{genus}"}})
        
        MERGE (p:Plant {{plant_name: "{plant_name}"}})
        SET p.scientific_name = "{scientific_name}",
            p.family = "{family}",
            p.kingdom = "{kingdom}",
            p.order = "{order}",
            p.genus = "{genus}",
            p.species = "{species}",
            p.medicinal_properties = "{medicinal_properties}",
            p.habitat = "{habitat}",
            p.uses = "{uses}",
            p.chemical_components = "{chemical_components}",
            p.created_at = datetime(),
            p.auto_generated = true
        
        MERGE (p)-[:BELONGS_TO_FAMILY]->(f)
        MERGE (p)-[:BELONGS_TO_GENUS]->(g)
        MERGE (f)-[:IN_KINGDOM]->(k)
        MERGE (g)-[:IN_FAMILY]->(f)
        '''
        
        if medicinal_properties and medicinal_properties.strip():
            cypher += f'''
            WITH p
            MERGE (mp:MedicinalProperty {{description: "{medicinal_properties[:500]}"}})
            MERGE (p)-[:HAS_MEDICINAL_PROPERTY]->(mp)
            '''
        
        return cypher
    
    def insert_plant_data(self, cypher_query: str) -> Tuple[bool, str]:
        """Execute enhanced Cypher CREATE query"""
        if not self.driver:
            return False, "No database connection available"
            
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query)
                summary = result.consume()
                return True, f"Successfully created {summary.counters.nodes_created} nodes and {summary.counters.relationships_created} relationships"
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower() or "constraint" in error_msg.lower():
                    return True, "Plant already exists in database"
                return False, f"Error inserting data: {error_msg}"
    
    
        """Load sample plant data"""
        sample_plants = [
            {
                'plant_name': 'Schleichera_Oleosa',
                'scientific_name': 'Schleichera oleosa',
                'family': 'Sapindaceae',
                'kingdom': 'Plantae',
                'order': 'Sapindales',
                'genus': 'Schleichera',
                'species': 'oleosa',
                'medicinal_properties': 'Used in traditional medicine for skin diseases, wounds, and digestive issues',
                'habitat': 'Tropical forests of India and Southeast Asia',
                'uses': 'Oil extraction, timber, traditional medicine',
                'chemical_components': 'Saponins, tannins, essential oils'
            },
            {
                'plant_name': 'Turmeric',
                'scientific_name': 'Curcuma longa',
                'family': 'Zingiberaceae',
                'kingdom': 'Plantae',
                'order': 'Zingiberales',
                'genus': 'Curcuma',
                'species': 'longa',
                'medicinal_properties': 'Anti-inflammatory, antioxidant, antimicrobial properties',
                'habitat': 'Native to Southeast Asia, cultivated in tropical regions',
                'uses': 'Culinary spice, traditional medicine, cosmetics',
                'chemical_components': 'Curcumin, essential oils, proteins'
            },
            {
                'plant_name': 'Neem',
                'scientific_name': 'Azadirachta indica',
                'family': 'Meliaceae',
                'kingdom': 'Plantae',
                'order': 'Sapindales',
                'genus': 'Azadirachta',
                'species': 'indica',
                'medicinal_properties': 'Antimicrobial, antifungal, anti-inflammatory, immunomodulatory',
                'habitat': 'Native to Indian subcontinent, grown in tropical regions',
                'uses': 'Traditional medicine, pesticide, cosmetics, timber',
                'chemical_components': 'Azadirachtin, nimbin, nimbidin, quercetin'
            }
        ]
        
        success_count = 0
        error_count = 0
        
        for plant_data in sample_plants:
            try:
                cypher_query = self.template_enhanced_cypher_insert(plant_data)
                success, message = self.insert_plant_data(cypher_query)
                
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
        
        if success_count > 0:
            self.data_loaded = True
        
        return success_count, error_count
    
    def template_enhanced_cypher_query(self, plant_name: str) -> str:
        """Enhanced template query with relationships"""
        plant_name_clean = plant_name.replace('"', '\\"').strip()
        
        cypher = f'''
        MATCH (p:Plant)
        WHERE toLower(p.plant_name) CONTAINS toLower("{plant_name_clean}")
           OR toLower(p.scientific_name) CONTAINS toLower("{plant_name_clean}")
        
        OPTIONAL MATCH (p)-[:BELONGS_TO_FAMILY]->(f:Family)
        OPTIONAL MATCH (p)-[:BELONGS_TO_GENUS]->(g:Genus)
        OPTIONAL MATCH (p)-[:HAS_MEDICINAL_PROPERTY]->(mp:MedicinalProperty)
        OPTIONAL MATCH (f)<-[:BELONGS_TO_FAMILY]-(related:Plant)
        WHERE related <> p
        
        RETURN p.plant_name as plant_name,
               p.scientific_name as scientific_name,
               p.family as family,
               p.kingdom as kingdom,
               p.genus as genus,
               p.species as species,
               p.medicinal_properties as medicinal_properties,
               p.habitat as habitat,
               p.uses as uses,
               p.chemical_components as chemical_components,
               p.auto_generated as auto_generated,
               f.name as family_name,
               g.name as genus_name,
               collect(DISTINCT related.plant_name)[0..3] as related_plants,
               collect(DISTINCT mp.description)[0..2] as medicinal_properties_list
        LIMIT 5
        '''
        
        return cypher
    
    def query_plant_data(self, cypher_query: str) -> Tuple[bool, List[Dict]]:
        """Execute enhanced query"""
        if not self.driver:
            return False, "No database connection available"
            
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return True, records
            except Exception as e:
                return False, f"Error querying data: {str(e)}"
    
    def search_plants(self, query: str) -> Tuple[bool, List[Dict], str]:
        """Search for plants and return results"""
        cypher_query = self.template_enhanced_cypher_query(query)
        success, results = self.query_plant_data(cypher_query)
        
        if success and results:
            response = self.format_search_results(results, query)
            return True, results, response
        else:
            error_msg = results if not success else "No matches found"
            return False, [], f"Search failed: {error_msg}"
    
    def format_search_results(self, plant_data_list: List[Dict], original_query: str) -> str:
        """Format search results for display"""
        if not plant_data_list:
            return f"No plant information found for '{original_query}'"
        
        response = f"Plant Knowledge Graph Results for '{original_query}'\n\n"
        response += f"Found {len(plant_data_list)} matching plants:\n\n"
        
        for i, plant in enumerate(plant_data_list, 1):
            auto_gen_marker = " 🤖" if plant.get('auto_generated') else ""
            response += f"{i}. {plant.get('plant_name', 'Unknown Plant')}{auto_gen_marker}\n"
            
            if plant.get('scientific_name'):
                response += f"   Scientific Name: {plant.get('scientific_name')}\n"
            
            if plant.get('family'):
                response += f"   Family: {plant.get('family')}\n"
            
            if plant.get('medicinal_properties'):
                med_props = plant.get('medicinal_properties')[:200]
                response += f"   Medicinal Properties: {med_props}{'...' if len(plant.get('medicinal_properties', '')) > 200 else ''}\n"
            
            if plant.get('uses'):
                uses = plant.get('uses')[:200]
                response += f"   Uses: {uses}{'...' if len(plant.get('uses', '')) > 200 else ''}\n"
            
            response += "\n"
        
        return response

# Initialize Enhanced Knowledge Graph
kg = EnhancedPlantKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, MODEL_CHOICE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Home page showing KG status"""
    connection_status = "Connected" if kg.connection_tested else "Not Tested"
    data_status = "Loaded" if kg.data_loaded else "Not Loaded"
    
    return jsonify({
        "message": "Enhanced Plant Knowledge Graph API with Auto-Generation",
        "connection_status": connection_status,
        "data_status": data_status,
        "features": [
            "🔍 Smart plant search with auto-generation",
            "📸 Plant identification from images",
            "🤖 AI-powered data generation",
            "🌐 Web scraping for plant data",
            "📊 Knowledge graph relationships"
        ],
        "endpoints": {
            "test_connection": "/test_connection",
            "search": "/search/<plant_name>",
            "smart_search": "/smart_search/<plant_name>",
            "predict": "/predict (upload image or search by name)",
            "identify_image": "/identify_image (upload plant image)"
        }
    })

@app.route('/status')
def status():
    """Enhanced API status endpoint"""
    if kg.models_loading:
        ai_models_status = "Loading in background..."
    elif kg.models_loaded:
        ai_models_status = "Loaded"
    else:
        ai_models_status = "Fallback Mode"
    
    return jsonify({
        "kg_available": True,
        "connection_tested": kg.connection_tested,
        "data_loaded": kg.data_loaded,
        "ai_models": ai_models_status,
        "models_loading": kg.models_loading,
        "models_loaded": kg.models_loaded,
        "features": {
            "image_recognition": kg.models_loaded and bool(kg.image_model),
            "text_generation": kg.models_loaded and bool(kg.text_generator),
            "web_scraping": True,
            "auto_generation": True
        },
        "neo4j_uri": NEO4J_URI.split('@')[1] if '@' in NEO4J_URI else "configured"
    })

@app.route('/ai_status')
def ai_status():
    """Check AI models status"""
    return jsonify({
        'loading_status': {
            'models_loading': kg.models_loading,
            'models_loaded': kg.models_loaded
        },
        'ai_models': {
            'image_recognition': {
                'loaded': kg.models_loaded and bool(kg.image_model and kg.image_processor),
                'model': 'Salesforce/blip-image-captioning-base' if kg.models_loaded and kg.image_model else 'Loading...' if kg.models_loading else 'Not loaded'
            },
            'text_generation': {
                'loaded': kg.models_loaded and bool(kg.text_generator),
                'model': 'microsoft/DialoGPT-medium' if kg.models_loaded and kg.text_generator else 'Loading...' if kg.models_loading else 'Not loaded'
            }
        },
        'fallback_methods': {
            'web_apis': 'GBIF, Wikipedia APIs available',
            'template_generation': 'Available for all plants'
        },
        'capabilities': [
            '🔍 Plant search in existing knowledge graph',
            '🤖 Auto-generation of missing plant data',
            '📸 Plant identification from images' + (' (loading...)' if kg.models_loading else ' (ready)' if kg.models_loaded else ' (fallback mode)'),
            '🌐 Web scraping from botanical APIs',
            '📊 Relationship mapping in knowledge graph'
        ]
    })
@app.route('/test_connection')
def test_database_connection():
    """Test database connection endpoint with detailed diagnostics"""
    try:
        if not kg.driver:
            return jsonify({
                "success": False,
                "message": f"No database driver available. Last error: {kg.last_error}",
                "connection_tested": False,
                "diagnostic_info": {
                    "neo4j_uri": kg.uri,
                    "username": kg.username,
                    "password_set": bool(kg.password)
                }
            }), 500
        
        success, message = kg.test_connection()
        
        response_data = {
            "success": success,
            "message": message,
            "connection_tested": kg.connection_tested,
            "current_uri": kg.uri,
            "diagnostic_info": {
                "driver_available": bool(kg.driver),
                "last_error": kg.last_error
            }
        }
        
        if not success:
            response_data["troubleshooting"] = {
                "suggestions": [
                    "Check if Neo4j database is running",
                    "Verify credentials are correct",
                    "Check network connectivity",
                    "Try different URI formats"
                ],
            }
            return jsonify(response_data), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Connection test failed with exception: {str(e)}",
            "connection_tested": False,
            "error_type": type(e).__name__
        }), 500


    """Load sample plant data endpoint"""
    try:
        success_count, error_count = kg.load_sample_data()
        return jsonify({
            "success": success_count > 0,
            "message": f"Successfully loaded {success_count} plants, {error_count} errors",
            "success_count": success_count,
            "error_count": error_count,
            "data_loaded": kg.data_loaded
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Data loading failed: {str(e)}",
            "success_count": 0,
            "error_count": 1
        }), 500

@app.route('/search/<plant_name>')
def search_plants(plant_name):
    """Basic search plants endpoint (existing data only)"""
    try:
        success, results, response = kg.search_plants(plant_name)
        
        return jsonify({
            "success": success,
            "query": plant_name,
            "results_count": len(results),
            "results": results,
            "formatted_response": response,
            "search_type": "existing_data_only"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "query": plant_name,
            "message": f"Search failed: {str(e)}",
            "results_count": 0,
            "results": []
        }), 500

@app.route('/smart_search/<plant_name>')
def smart_search_plants(plant_name):
    """Smart search with auto-generation capability"""
    try:
        success, results, response = kg.search_or_generate_plant_data(plant_name)
        
        return jsonify({
            "success": success,
            "query": plant_name,
            "results_count": len(results),
            "results": results,
            "formatted_response": response,
            "search_type": "smart_search_with_generation"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "query": plant_name,
            "message": f"Smart search failed: {str(e)}",
            "results_count": 0,
            "results": []
        }), 500

@app.route('/identify_image', methods=['POST'])
@cross_origin()
def identify_plant_image():
    """Plant identification from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Identify plant from image
            identified_plant, confidence = kg.identify_plant_from_image(file_path)
            
            # Search or generate data for identified plant
            success, results, response = kg.search_or_generate_plant_data(identified_plant)
            
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify({
                'success': True,
                'identified_plant': identified_plant,
                'confidence': confidence,
                'results_count': len(results),
                'results': results,
                'formatted_response': response,
                'filename': unique_filename
            }), 200
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': f"Image identification failed: {str(e)}"}), 500

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    """Enhanced plant identification endpoint"""
    try:
        if request.method == 'GET':
            return jsonify({
                "message": "Enhanced Plant identification endpoint",
                "capabilities": [
                    "📸 Upload image for plant identification",
                    "🔍 Search by plant name with auto-generation",
                    "🤖 AI-powered data generation for new plants",
                    "🌐 Web scraping for comprehensive plant data"
                ],
                "usage": {
                    "image_upload": "POST with 'file' in form-data",
                    "text_search": "POST with JSON {'plant_name': 'name'}"
                }
            })

        # Handle file upload for image-based identification
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                
                # Identify plant from image
                identified_plant, confidence = kg.identify_plant_from_image(file_path)
                
                # Get comprehensive data
                success, results, response = kg.search_or_generate_plant_data(identified_plant)
                
                # Clean up
                try:
                    os.remove(file_path)
                except:
                    pass
                
                if success and results:
                    return jsonify({
                        'method': 'image_identification',
                        'identified_plant': identified_plant,
                        'identification_confidence': confidence,
                        'species': results[0].get('plant_name', 'Unknown'),
                        'scientific_name': results[0].get('scientific_name', 'Unknown'),
                        'family': results[0].get('family', 'Unknown'),
                        'medicinal_properties': results[0].get('medicinal_properties', ''),
                        'uses': results[0].get('uses', ''),
                        'habitat': results[0].get('habitat', ''),
                        'chemical_components': results[0].get('chemical_components', ''),
                        'auto_generated': results[0].get('auto_generated', False),
                        'total_matches': len(results),
                        'all_results': results,
                        'filename': unique_filename
                    }), 200
                else:
                    return jsonify({'error': f'Could not identify or generate data for plant in image'}), 404

        # Handle text-based search with JSON
        elif request.json and 'plant_name' in request.json:
            plant_name = request.json['plant_name']
            success, results, response = kg.search_or_generate_plant_data(plant_name)
            
            if success and results:
                return jsonify({
                    'method': 'text_search_with_generation',
                    'species': results[0].get('plant_name', 'Unknown'),
                    'scientific_name': results[0].get('scientific_name', 'Unknown'),
                    'family': results[0].get('family', 'Unknown'),
                    'confidence': 0.95,  # High confidence for exact name searches
                    'medicinal_properties': results[0].get('medicinal_properties', ''),
                    'uses': results[0].get('uses', ''),
                    'habitat': results[0].get('habitat', ''),
                    'chemical_components': results[0].get('chemical_components', ''),
                    'auto_generated': results[0].get('auto_generated', False),
                    'total_matches': len(results),
                    'all_results': results
                }), 200
            else:
                return jsonify({'error': f'Could not find or generate data for: {plant_name}'}), 404

        return jsonify({'error': 'No file or plant_name provided'}), 400

    except Exception as e:
        return jsonify({'error': f"Internal Server Error: {str(e)}"}), 500

@app.route('/generate_plant_data', methods=['POST'])
def generate_plant_data():
    """Manually trigger plant data generation"""
    try:
        data = request.get_json()
        if not data or 'plant_name' not in data:
            return jsonify({'error': 'plant_name required in JSON body'}), 400
        
        plant_name = data['plant_name']
        
        # Generate comprehensive plant data
        plant_data = kg.generate_plant_data_from_web(plant_name)
        
        # Optionally save to database
        save_to_db = data.get('save_to_database', True)
        if save_to_db:
            cypher_query = kg.template_enhanced_cypher_insert(plant_data)
            success, message = kg.insert_plant_data(cypher_query)
            
            return jsonify({
                'success': success,
                'plant_name': plant_name,
                'generated_data': plant_data,
                'database_save': {
                    'success': success,
                    'message': message
                }
            })
        else:
            return jsonify({
                'success': True,
                'plant_name': plant_name,
                'generated_data': plant_data,
                'database_save': {'success': False, 'message': 'Not saved (save_to_database=False)'}
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Data generation failed: {str(e)}"
        }), 500

@app.route('/quick_demo')
def quick_demo():
    """Enhanced quick demo with auto-generation"""
    try:
        results = {}
        
        # 1. Test connection
        success, message = kg.test_connection()
        results['connection_test'] = {
            'success': success,
            'message': message
        }
        
        if success:
            # 2. Load sample data
            success_count, error_count = kg.load_sample_data()
            results['data_loading'] = {
                'success': success_count > 0,
                'success_count': success_count,
                'error_count': error_count
            }
            
            # 3. Test existing plant search
            search_success, search_results, search_response = kg.search_plants('Turmeric')
            results['existing_plant_search'] = {
                'success': search_success,
                'query': 'Turmeric',
                'results_count': len(search_results),
                'results': search_results
            }
            
            # 4. Test smart search with auto-generation for new plant
            smart_success, smart_results, smart_response = kg.search_or_generate_plant_data('Lavender')
            results['smart_search_demo'] = {
                'success': smart_success,
                'query': 'Lavender',
                'results_count': len(smart_results),
                'results': smart_results,
                'auto_generated': smart_results[0].get('auto_generated', False) if smart_results else False
            }
        
        return jsonify({
            'demo_completed': True,
            'results': results,
            'ai_models_loaded': {
                'image_model': bool(kg.image_model),
                'text_generator': bool(kg.text_generator)
            }
        })
        
    except Exception as e:
        return jsonify({
            'demo_completed': False,
            'error': str(e)
        }), 500

@app.route('/ai_status')
def ai_status():
    """Check AI models status"""
    return jsonify({
        'ai_models': {
            'image_recognition': {
                'loaded': bool(kg.image_model and kg.image_processor),
                'model': 'Salesforce/blip-image-captioning-base' if kg.image_model else 'Not loaded'
            },
            'text_generation': {
                'loaded': bool(kg.text_generator),
                'model': 'microsoft/DialoGPT-medium' if kg.text_generator else 'Not loaded'
            }
        },
        'fallback_methods': {
            'web_apis': 'GBIF, Wikipedia APIs available',
            'template_generation': 'Available for all plants'
        },
        'capabilities': [
            '🔍 Plant search in existing knowledge graph',
            '🤖 Auto-generation of missing plant data',
            '📸 Plant identification from images',
            '🌐 Web scraping from botanical APIs',
            '📊 Relationship mapping in knowledge graph'
        ]
    })

if __name__ == "__main__":
    print("🌿 Starting Enhanced Plant Knowledge Graph Flask App...")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Port: {PORT}")
    print("\n🤖 AI Features:")
    print("- Plant identification from images")
    print("- Auto-generation of missing plant data")
    print("- Web scraping from botanical APIs")
    print("- Smart search with fallback generation")
    print("\nAvailable endpoints:")
    print("- GET  /                          - Home/API info")
    print("- GET  /status                    - System status")  
    print("- GET  /ai_status                 - AI models status")
    print("- GET  /test_connection           - Test DB connection")
    print("- GET  /search/<plant_name>       - Search existing plants only")
    print("- GET  /smart_search/<plant_name> - Smart search with auto-generation")
    print("- POST /predict                   - Enhanced plant identification")
    print("- POST /identify_image            - Plant identification from image")
    print("- POST /generate_plant_data       - Manual data generation")
    print("- GET  /quick_demo                - Enhanced demo with AI features")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)