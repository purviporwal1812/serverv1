from dotenv import load_dotenv
load_dotenv()

import os, uuid, json, warnings, re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pymongo import MongoClient
import time
import requests
from PIL import Image
import io
import threading
import base64
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from neo4j import GraphDatabase

# **MUST** import TensorFlow here
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Read critical configuration
KERAS_MODEL_PATH       = os.path.join("models", "plant_model.tflite")  # Changed from TFLITE_MODEL_PATH
CLASS_INDICES_PATH     = os.path.join("models", "class_indices.json")
UPLOAD_FOLDER          = os.environ.get("UPLOAD_FOLDER", "uploads")
ALLOWED_EXTENSIONS     = set(os.environ.get("ALLOWED_EXTENSIONS", "png,jpg,jpeg,gif").split(","))
IMAGE_SIZE_STR         = os.environ.get("IMAGE_SIZE", "224,224")
PORT                   = int(os.environ.get("PORT", "10000"))

# Parse IMAGE_SIZE once
try:
    IMAGE_SIZE = tuple(int(x.strip()) for x in IMAGE_SIZE_STR.split(","))
    if len(IMAGE_SIZE) != 2:
        raise ValueError()
except Exception:
    raise ValueError("IMAGE_SIZE environment variable must be in the format 'width,height'")

# Initialize Flask
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    tflite_interpreter = tf.lite.Interpreter(model_path=KERAS_MODEL_PATH)  # Change variable name to TFLITE_MODEL_PATH
    tflite_interpreter.allocate_tensors()
    print(f"✅ TFLite model loaded successfully from {KERAS_MODEL_PATH}")
    
    # Print model info for debugging
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    print(f"📋 Input shape: {input_details[0]['shape']}")
    print(f"📋 Input dtype: {input_details[0]['dtype']}")
    print(f"📋 Output shape: {output_details[0]['shape']}")
    print(f"📋 Output dtype: {output_details[0]['dtype']}")
    
except Exception as e:
    print(f"❌ Failed to load TFLite model: {e}")
    tflite_interpreter = None
def make_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        else:
            return obj

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )

# Remove duplicate variable declarations
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

# Plant Knowledge Graph Class with Enhanced Capabilities
class EnhancedPlantKnowledgeGraph:
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        self.connection_tested = False
        self.data_loaded = False
        self.last_error = None
            
        # Try to establish connection with multiple URI formats
        self._establish_connection()
    
    def fetch_plant_data_from_apis(self, plant_name: str) -> Dict:
        """Fetch plant data from legitimate APIs like GBIF and Wikipedia"""
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
        
        try:
            # Try Wikipedia API first
            wikipedia_data = self.fetch_from_wikipedia(plant_name)
            if wikipedia_data:
                plant_data.update(wikipedia_data)
                print(f"✅ Retrieved data from Wikipedia for {plant_name}")
                return plant_data
            
            # Try GBIF API
            gbif_data = self.fetch_from_gbif(plant_name)
            if gbif_data:
                plant_data.update(gbif_data)
                print(f"✅ Retrieved data from GBIF for {plant_name}")
                return plant_data
            
            # Try other botanical APIs
            other_data = self.fetch_from_other_apis(plant_name)
            if other_data:
                plant_data.update(other_data)
                print(f"✅ Retrieved data from other APIs for {plant_name}")
                return plant_data
                
        except Exception as e:
            print(f"❌ API fetch failed for {plant_name}: {e}")
        
        return plant_data
    
    def fetch_from_wikipedia(self, plant_name: str) -> Dict:
        """Fetch plant information from Wikipedia API"""
        try:
            # Search for the plant page
            search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + plant_name.replace(" ", "_")
            
            headers = {
                'User-Agent': 'PlantKnowledgeGraph/1.0 (Educational Purpose)'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                extract = data.get('extract', '')
                
                plant_info = {}
                
                # Try to extract scientific name
                scientific_match = re.search(r'\b([A-Z][a-z]+ [a-z]+)\b', extract)
                if scientific_match:
                    plant_info['scientific_name'] = scientific_match.group(1)
                
                # Extract family information
                family_match = re.search(r'family\s+([A-Z][a-z]+aceae|[A-Z][a-z]+)', extract, re.IGNORECASE)
                if family_match:
                    plant_info['family'] = family_match.group(1)
                
                # Extract basic information for uses and properties
                if 'medicinal' in extract.lower() or 'medicine' in extract.lower():
                    plant_info['medicinal_properties'] = 'Has traditional medicinal uses'
                
                if extract:
                    # Use first 200 characters as description
                    plant_info['uses'] = extract[:200] + "..." if len(extract) > 200 else extract
                
                return plant_info
                
        except Exception as e:
            print(f"Wikipedia API error for {plant_name}: {e}")
        
        return {}
    
    def fetch_from_gbif(self, plant_name: str) -> Dict:
        """Fetch plant information from GBIF (Global Biodiversity Information Facility)"""
        try:
            # Search GBIF species API
            search_url = f"https://api.gbif.org/v1/species/search"
            params = {
                'q': plant_name,
                'limit': 5,
                'kingdom': 'Plantae'
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                if results:
                    # Take the first matching result
                    species_data = results[0]
                    
                    plant_info = {}
                    
                    if species_data.get('scientificName'):
                        plant_info['scientific_name'] = species_data['scientificName']
                    
                    if species_data.get('family'):
                        plant_info['family'] = species_data['family']
                    
                    if species_data.get('order'):
                        plant_info['order'] = species_data['order']
                    
                    if species_data.get('genus'):
                        plant_info['genus'] = species_data['genus']
                    
                    if species_data.get('kingdom'):
                        plant_info['kingdom'] = species_data['kingdom']
                    
                    # Get additional details if available
                    species_key = species_data.get('key')
                    if species_key:
                        detail_info = self.fetch_gbif_species_detail(species_key)
                        if detail_info:
                            plant_info.update(detail_info)
                    
                    return plant_info
                    
        except Exception as e:
            print(f"GBIF API error for {plant_name}: {e}")
        
        return {}
    
    def fetch_gbif_species_detail(self, species_key: str) -> Dict:
        """Fetch detailed species information from GBIF"""
        try:
            detail_url = f"https://api.gbif.org/v1/species/{species_key}"
            
            response = requests.get(detail_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                detail_info = {}
                
                # Extract habitat information
                habitats = data.get('habitats', [])
                if habitats:
                    detail_info['habitat'] = ', '.join(habitats[:3])
                
                # Extract usage information if available
                if data.get('descriptions'):
                    descriptions = data['descriptions'][:2]  # Take first 2 descriptions
                    detail_info['uses'] = ' '.join([desc.get('description', '') for desc in descriptions])[:300]
                
                return detail_info
                
        except Exception as e:
            print(f"GBIF detail API error for species {species_key}: {e}")
        
        return {}
    
    def fetch_from_other_apis(self, plant_name: str) -> Dict:
        """Fetch from other free botanical APIs"""
        try:
            # Try Tropicos API (Missouri Botanical Garden)
            tropicos_data = self.fetch_from_tropicos(plant_name)
            if tropicos_data:
                return tropicos_data
            
        except Exception as e:
            print(f"Other APIs error for {plant_name}: {e}")
        
        return {}
    
    def fetch_from_tropicos(self, plant_name: str) -> Dict:
        """Fetch from Tropicos API (Missouri Botanical Garden)"""
        try:
            # Note: Tropicos requires API key for full access, using basic search
            search_url = f"http://services.tropicos.org/Name/Search"
            params = {
                'name': plant_name,
                'format': 'json'
            }
            
            response = requests.get(search_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    species_info = data[0]
                    
                    plant_info = {}
                    
                    if species_info.get('ScientificName'):
                        plant_info['scientific_name'] = species_info['ScientificName']
                    
                    if species_info.get('Family'):
                        plant_info['family'] = species_info['Family']
                    
                    return plant_info
                    
        except Exception as e:
            print(f"Tropicos API error for {plant_name}: {e}")
        
        return {}

    def generate_plant_data_from_web(self, plant_name: str) -> Dict:
        """Generate comprehensive plant data using web APIs"""
        try:
            # Fetch from legitimate APIs
            plant_data = self.fetch_plant_data_from_apis(plant_name)
            
            return plant_data
            
        except Exception as e:
            print(f"❌ Data generation failed for {plant_name}: {e}")
            return self.create_minimal_plant_data(plant_name)
    
    def create_minimal_plant_data(self, plant_name: str) -> Dict:
        """Create minimal plant data structure when APIs fail"""
        return {
            'plant_name': plant_name,
            'scientific_name': 'Unknown',
            'family': 'Unknown Family',
            'kingdom': 'Plantae',
            'order': 'Unknown Order',
            'genus': plant_name.split()[0] if ' ' in plant_name else 'Unknown',
            'species': plant_name.split()[1] if len(plant_name.split()) > 1 else 'Unknown',
            'medicinal_properties': 'Information not available',
            'habitat': 'Various environments',
            'uses': 'Uses to be researched',
            'chemical_components': 'Components under study'
        }
 
    def predict_species(self, image_path, class_indices_path=None):
        """Predict plant species using TFLite model with EfficientNet-style preprocessing
        Mirrors the Kaggle pipeline: PIL -> resize -> efficientnet.preprocess_input -> tflite.invoke
        """
        if class_indices_path is None:
            class_indices_path = CLASS_INDICES_PATH

        # Sanity checks
        if not os.path.exists(class_indices_path):
            return {"error": "Class indices file not available. Please check server logs."}
        if tflite_interpreter is None:
            return {"error": "TFLite interpreter not loaded. Please check server logs."}

        # Load class mapping and create idx->name mapping like on Kaggle
        try:
            with open(class_indices_path, "r") as f:
                class_map = json.load(f)

            # If the file is name->idx (common), invert to idx->name
            # If it's idx->name, just coerce keys to int
            if all(not str(k).isdigit() for k in class_map.keys()):
                # name -> idx  (values may be int or str)
                idx_to_name = {int(v): str(k) for k, v in class_map.items()}
            else:
                # idx -> name
                idx_to_name = {int(k): str(v) for k, v in class_map.items()}
        except Exception as e:
            return {"error": f"Could not load/parse class indices: {str(e)}"}

        # Load & preprocess image using same pipeline as Kaggle (EfficientNet preprocess_input)
        try:
            from PIL import Image
            img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.BILINEAR)
            img_arr = np.array(img).astype(np.float32)

            # Prefer EfficientNet preprocess_input (same as your Kaggle notebook)
            try:
                from tensorflow.keras.applications.efficientnet import preprocess_input
                img_arr = preprocess_input(img_arr)
                preprocess_used = "efficientnet.preprocess_input"
            except Exception:
                # Fallback to generic imagenet preprocess_input if available
                try:
                    from tensorflow.keras.applications.imagenet_utils import preprocess_input
                    img_arr = preprocess_input(img_arr)
                    preprocess_used = "imagenet.preprocess_input"
                except Exception:
                    # Last resort: simple scaling [0,1]
                    img_arr = img_arr / 255.0
                    preprocess_used = "scaled_0_1"

            img_batch = np.expand_dims(img_arr, axis=0)
        except Exception as e:
            return {"error": f"Could not process image: {str(e)}"}

        # Predict using TFLite interpreter (match Kaggle behaviour)
        try:
            # Get input and output tensor details
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            
            # Ensure input data type matches model requirements
            input_dtype = input_details[0]['dtype']
            if input_dtype == np.uint8:
                # Model expects uint8 input (0-255 range)
                if img_arr.min() >= 0 and img_arr.max() <= 1:
                    # Data is normalized [0,1], scale to [0,255]
                    img_batch = (img_batch * 255.0).astype(np.uint8)
                else:
                    # Data might already be in correct range or preprocessed differently
                    img_batch = img_batch.astype(np.uint8)
            elif input_dtype == np.float32:
                # Model expects float32 input (already correct)
                img_batch = img_batch.astype(np.float32)
            else:
                # Handle other data types if needed
                img_batch = img_batch.astype(input_dtype)
            
            # Verify input shape matches model requirements
            expected_shape = input_details[0]['shape']
            if img_batch.shape != tuple(expected_shape):
                return {"error": f"Input shape mismatch. Expected: {expected_shape}, Got: {img_batch.shape}"}
            
            # Set input tensor
            tflite_interpreter.set_tensor(input_details[0]['index'], img_batch)
            
            # Run inference
            tflite_interpreter.invoke()
            
            # Get output tensor
            preds = tflite_interpreter.get_tensor(output_details[0]['index'])

            # If model returns multiple outputs, assume first is logits/probs
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            preds = np.asarray(preds)
            # If batch dimension present, reduce to first batch
            if preds.ndim == 2 and preds.shape[0] == 1:
                preds = preds[0]
            elif preds.ndim > 2:
                # Unexpected shape
                preds = preds.reshape((preds.shape[0], -1))
                if preds.shape[0] == 1:
                    preds = preds[0]

            # Now preds should be 1D array of logits/probs
            if preds.ndim != 1:
                return {"error": f"Unexpected prediction shape: {preds.shape}"}

            # Same as your Kaggle code: argmax + max
            predicted_idx = int(np.argmax(preds))
            confidence = float(np.max(preds))

            # If outputs are logits (not probabilities), confidence may not be in [0,1].
            # But we return exactly as your Kaggle snippet does; if you want probabilities,
            # set apply_softmax=True below and it'll convert logits -> softmax probs.
            apply_softmax = False

            probs = preds
            if apply_softmax:
                try:
                    import tensorflow as _tf
                    probs = _tf.nn.softmax(preds).numpy()
                except Exception:
                    e_x = np.exp(preds - np.max(preds))
                    probs = e_x / e_x.sum()
                predicted_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))

            # Top-5 (or fewer if classes < 5)
            top_k = min(5, probs.shape[0])
            top_idxs = np.argsort(-probs)[:top_k]

            top_predictions = [
                {"species": idx_to_name.get(int(i), str(int(i))), "confidence": float(probs[int(i)])}
                for i in top_idxs
            ]

            result = {
                "species": idx_to_name.get(predicted_idx, str(predicted_idx)),
                "confidence": confidence,
                "top_predictions": top_predictions,
                "preprocess_used": preprocess_used,
                "raw_prediction_shape": list(preds.shape),  # Convert to list for JSON serialization
                "model_type": "TFLite",
                "input_dtype": str(input_dtype),
                "expected_input_shape": list(expected_shape)  # Convert to list for JSON serialization
            }
            result = make_json_serializable(result)

            return result
        except Exception as e:
            return {"error": f"TFLite prediction failed: {str(e)}"}

    
    def search_or_generate_plant_data(self, plant_name: str) -> Tuple[bool, List[Dict], str]:
        """Search for plant data, generate if not found using legitimate APIs"""
        try:
            # First, try to search existing data
            success, results, response = self.search_plants(plant_name)
            
            if success and results:
                return success, results, response
            
            # If not found, generate new data from APIs
            print(f"🔄 Plant '{plant_name}' not found in KG. Fetching from APIs...")
            
            # Generate comprehensive plant data from APIs
            new_plant_data = self.generate_plant_data_from_web(plant_name)
            
            # Insert into knowledge graph
            cypher_query = self.template_enhanced_cypher_insert(new_plant_data)
            insert_success, insert_message = self.insert_plant_data(cypher_query)
            
            if insert_success:
                print(f"✅ Successfully added {plant_name} to knowledge graph")
                
                # Now search again to return the newly added data
                success, results, response = self.search_plants(plant_name)
                
                if success and results:
                    response = f"🆕 Generated and added new plant data for '{plant_name}' from APIs\n\n" + response
                    return True, results, response
            
            # If insertion failed, return the generated data anyway
            results = [new_plant_data]
            response = self.format_search_results(results, plant_name)
            response = f"🆕 Generated plant data for '{plant_name}' from APIs (not saved to database)\n\n" + response
            
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
            p.auto_generated = true,
            p.data_source = "APIs"
        
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
               p.data_source as data_source,
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
            auto_gen_marker = " 🌐" if plant.get('auto_generated') else ""
            data_source = plant.get('data_source', '')
            source_marker = f" ({data_source})" if data_source else ""
            
            response += f"{i}. {plant.get('plant_name', 'Unknown Plant')}{auto_gen_marker}{source_marker}\n"
            
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
kg = EnhancedPlantKnowledgeGraph(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Flask Routes - All Endpoints Implementation
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import os, uuid, requests

# Initialize Flask App
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# CORE APPLICATION ENDPOINTS
# =============================================================================

@app.route('/')
def home():
    """Home page with API information and model status"""
    model_status = "Available" if os.path.exists(KERAS_MODEL_PATH) else "Not Available"  # Changed
    class_indices_status = "Available" if os.path.exists(CLASS_INDICES_PATH) else "Not Available"
    
    return render_template('index.html', 
                         model_status=model_status, 
                         class_indices_status=class_indices_status)

@app.route('/status')
def status():
    """Enhanced system status endpoint"""    
    return jsonify({
        "service": "Enhanced Plant Knowledge Graph API",
        "version": "2.0.0",
        "kg_available": True,
        "connection_tested": kg.connection_tested,
        "data_loaded": kg.data_loaded,
        "data_sources": "Wikipedia, GBIF, Tropicos APIs",
        "features": {
            "web_apis": True,
            "wikipedia_integration": True,
            "gbif_integration": True,
            "auto_generation": True,
            "image_classification": os.path.exists(KERAS_MODEL_PATH),
            "knowledge_graph": bool(kg.driver)
        },
        "neo4j_uri": NEO4J_URI.split('@')[1] if '@' in NEO4J_URI else "configured",
        "endpoints": {
            "search": "/search/<plant_name>",
            "smart_search": "/smart_search/<plant_name>",
            "predict": "/predict",
            "generate_data": "/generate_plant_data",
            "test_apis": "/test_apis",
        }
    })

# =============================================================================
# CONNECTION & TESTING ENDPOINTS
# =============================================================================

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
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diagnostic_info": {
                "driver_available": bool(kg.driver),
                "last_error": kg.last_error,
                "connection_type": "neo4j+s" if "neo4j+s" in kg.uri else "bolt"
            }
        }
        
        if not success:
            response_data["troubleshooting"] = {
                "suggestions": [
                    "Check if Neo4j database is running",
                    "Verify credentials are correct",
                    "Check network connectivity",
                    "Ensure firewall allows Neo4j ports",
                    "Try different URI formats"
                ],
                "common_issues": [
                    "Authentication failure",
                    "Network timeout",
                    "SSL certificate issues"
                ]
            }
            return jsonify(response_data), 500
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Connection test failed with exception: {str(e)}",
            "connection_tested": False,
            "error_type": type(e).__name__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/test_apis')
def test_apis():
    """Test individual API sources connectivity"""
    try:
        api_results = {}
        
        # Test Wikipedia API
        try:
            response = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/Rose", 
                                  headers={'User-Agent': 'PlantKnowledgeGraph/1.0'}, 
                                  timeout=10)
            api_results['wikipedia'] = {
                'status': 'Available' if response.status_code == 200 else 'Error',
                'response_code': response.status_code,
                'response_time': f"{response.elapsed.total_seconds():.2f}s",
                'url': 'https://en.wikipedia.org/api/rest_v1/',
                'description': 'Wikipedia REST API for plant summaries'
            }
        except Exception as e:
            api_results['wikipedia'] = {
                'status': 'Error',
                'error': str(e),
                'description': 'Wikipedia REST API for plant summaries'
            }
        
        # Test GBIF API
        try:
            response = requests.get("https://api.gbif.org/v1/species/search?q=Rose&limit=1", 
                                  timeout=10)
            api_results['gbif'] = {
                'status': 'Available' if response.status_code == 200 else 'Error',
                'response_code': response.status_code,
                'response_time': f"{response.elapsed.total_seconds():.2f}s",
                'url': 'https://api.gbif.org/v1/',
                'description': 'Global Biodiversity Information Facility'
            }
        except Exception as e:
            api_results['gbif'] = {
                'status': 'Error',
                'error': str(e),
                'description': 'Global Biodiversity Information Facility'
            }
        
        # Test Tropicos API
        try:
            response = requests.get("http://services.tropicos.org/Name/Search?name=Rose&format=json", 
                                  timeout=10)
            api_results['tropicos'] = {
                'status': 'Available' if response.status_code == 200 else 'Error',
                'response_code': response.status_code,
                'response_time': f"{response.elapsed.total_seconds():.2f}s",
                'url': 'http://services.tropicos.org/',
                'description': 'Missouri Botanical Garden database'
            }
        except Exception as e:
            api_results['tropicos'] = {
                'status': 'Error',
                'error': str(e),
                'description': 'Missouri Botanical Garden database'
            }
        
        # Calculate overall API health
        available_apis = sum(1 for api in api_results.values() if api['status'] == 'Available')
        total_apis = len(api_results)
        
        return jsonify({
            'api_test_completed': True,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary': {
                'available_apis': available_apis,
                'total_apis': total_apis,
                'health_percentage': f"{(available_apis/total_apis)*100:.1f}%"
            },
            'apis': api_results
        })
        
    except Exception as e:
        return jsonify({
            'api_test_completed': False,
            'error': str(e),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500


# =============================================================================
# API STATUS & INFORMATION ENDPOINTS
# =============================================================================

@app.route('/api_status')
def api_status():
    """Check API sources status and capabilities"""
    return jsonify({
        'service_info': {
            'name': 'Enhanced Plant Knowledge Graph',
            'version': '2.0.0',
            'description': 'Plant identification and knowledge graph with API integration'
        },
        'data_sources': {
            'wikipedia': {
                'available': True,
                'description': 'Wikipedia REST API for plant summaries',
                'endpoint': 'https://en.wikipedia.org/api/rest_v1/',
                'data_provided': ['scientific_name', 'family', 'basic_description', 'medicinal_uses']
            },
            'gbif': {
                'available': True,
                'description': 'Global Biodiversity Information Facility',
                'endpoint': 'https://api.gbif.org/v1/',
                'data_provided': ['taxonomic_classification', 'habitat', 'distribution', 'species_details']
            },
            'tropicos': {
                'available': True,
                'description': 'Missouri Botanical Garden database',
                'endpoint': 'http://services.tropicos.org/',
                'data_provided': ['scientific_name', 'family', 'nomenclature']
            }
        },
        'fallback_methods': {
            'minimal_data_generation': {
                'description': 'Available for all plants when APIs fail',
                'provides': 'Basic plant structure with unknown values'
            }
        },
        'features': {
            'image_classification': {
                'available': os.path.exists(KERAS_MODEL_PATH),
                'model': 'Keras',
                'supported_formats': ['png', 'jpg', 'jpeg', 'gif']
            },
            'knowledge_graph': {
                'available': bool(kg.driver),
                'database': 'Neo4j',
                'relationships': ['BELONGS_TO_FAMILY', 'BELONGS_TO_GENUS', 'HAS_MEDICINAL_PROPERTY']
            }
        },
        'capabilities': [
            '🔍 Plant search in existing knowledge graph',
            '🌐 Auto-generation from Wikipedia API',
            '📊 GBIF species database integration',
            '🌿 Tropicos botanical data',
            '🔗 Relationship mapping in knowledge graph',
            '📸 Image-based plant identification',
            '🤖 Smart search with fallback generation'
        ]
    })

# =============================================================================
# PLANT SEARCH ENDPOINTS
# =============================================================================

@app.route('/search/<plant_name>')
def search_plants(plant_name):
    """Basic search plants endpoint (existing data only)"""
    try:
        start_time = time.time()
        success, results, response = kg.search_plants(plant_name)
        search_time = time.time() - start_time
        
        return jsonify({
            "success": success,
            "query": plant_name,
            "results_count": len(results),
            "results": results,
            "formatted_response": response,
            "search_type": "existing_data_only",
            "search_time": f"{search_time:.3f}s",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "knowledge_graph"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "query": plant_name,
            "message": f"Search failed: {str(e)}",
            "results_count": 0,
            "results": [],
            "error_type": type(e).__name__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

@app.route('/smart_search/<plant_name>')
def smart_search_plants(plant_name):
    """Smart search with API-based auto-generation capability"""
    try:
        start_time = time.time()
        success, results, response = kg.search_or_generate_plant_data(plant_name)
        search_time = time.time() - start_time
        
        # Determine if data was generated
        was_generated = any(result.get('auto_generated', False) for result in results)
        data_sources_used = []
        
        if was_generated:
            data_sources_used = ['Wikipedia', 'GBIF', 'Tropicos']
        else:
            data_sources_used = ['Knowledge Graph']
        
        return jsonify({
            "success": success,
            "query": plant_name,
            "results_count": len(results),
            "results": results,
            "formatted_response": response,
            "search_type": "smart_search_with_api_generation",
            "was_generated": was_generated,
            "data_sources_used": data_sources_used,
            "search_time": f"{search_time:.3f}s",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "query": plant_name,
            "message": f"Smart search failed: {str(e)}",
            "results_count": 0,
            "results": [],
            "error_type": type(e).__name__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

# =============================================================================
# IMAGE PREDICTION ENDPOINTS
# =============================================================================

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    """Plant species prediction from uploaded image"""
    try:
        if request.method == 'GET':
            return jsonify({
                "message": "Predict endpoint is accessible",
                "method": "POST",
                "required": "file (image)",
                "supported_formats": list(ALLOWED_EXTENSIONS),
                "max_file_size": "10MB",
                "model_available": os.path.exists(KERAS_MODEL_PATH)
            })

        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part in request',
                'required': 'multipart/form-data with file field'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'hint': 'Please select an image file'
            }), 400

        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)

            # Process prediction
            start_time = time.time()
            prediction_result = kg.predict_species(image_path=file_path)
            prediction_time = time.time() - start_time


            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass

            if 'error' in prediction_result:
                return jsonify({
                    'error': prediction_result['error'],
                    'processing_time': f"{prediction_time:.3f}s",
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }), 500

            predicted_species = prediction_result['species']
            
            # Connect to MongoDB and fetch image URLs
            MONGODB_URI = "mongodb+srv://0801cs221134:shi1234@cluster0.q3mah8x.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            
            try:
                # Connect to MongoDB
                client = MongoClient(MONGODB_URI)
                db = client["plant_database"]  # Database name from the image
                collection = db["plant_images"]  # Collection name from the image
                
                # Format the species name to match the MongoDB format
                # Remove underscores and try different formats since names might vary
                species_name = predicted_species.replace('_', ' ')
                
                # Create a regex pattern for flexible matching
                # This will match the species name regardless of spaces, underscores, or case
                pattern = re.compile(f"^{species_name.replace(' ', '[ _]')}$", re.IGNORECASE)
                
                # Try exact match first
                plant_data = collection.find_one({"plant_name": predicted_species})
                
                # If not found, try with spaces instead of underscores
                if not plant_data:
                    plant_data = collection.find_one({"plant_name": species_name})
                
                # If still not found, try with regex pattern
                if not plant_data:
                    plant_data = collection.find_one({"plant_name": {"$regex": pattern}})
                
                # If found, add image URLs to the result
                if plant_data and 'image_urls' in plant_data:
                    prediction_result['db_image_urls'] = plant_data['image_urls']
                else:
                    prediction_result['db_image_urls'] = []
                    prediction_result['db_note'] = "No matching plant found in database"
                
                # Check all top predictions for matches
                db_matches = []
                for pred in prediction_result['top_predictions']:
                    species = pred['species'].replace('_', ' ')
                    match = collection.find_one({"plant_name": {"$regex": f".*{species}.*", "$options": "i"}})
                    if match:
                        db_matches.append({
                            "species": pred['species'],
                            "confidence": pred['confidence'],
                            "db_match": match['plant_name'],
                            "image_urls": match.get('image_urls', [])
                        })
                
                prediction_result['db_matches'] = db_matches
                
            except Exception as db_error:
                prediction_result['db_error'] = str(db_error)
                prediction_result['db_error_type'] = type(db_error).__name__
            
            # Add metadata to result
            prediction_result['processing_time'] = f"{prediction_time:.3f}s"
            prediction_result['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
            prediction_result['image_size'] = list(IMAGE_SIZE)
            
            return jsonify(prediction_result), 200

        return jsonify({
            'error': 'Invalid file type',
            'supported_formats': list(ALLOWED_EXTENSIONS)
        }), 400

    except Exception as e:
        return jsonify({
            'error': f"Internal Server Error: {str(e)}",
            'error_type': type(e).__name__,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

# =============================================================================
# DATA GENERATION ENDPOINTS
# =============================================================================

@app.route('/generate_plant_data', methods=['POST'])
def generate_plant_data():
    """Manually trigger plant data generation from APIs"""
    try:
        # Validate request data
        data = request.get_json()
        if not data or 'plant_name' not in data:
            return jsonify({
                'error': 'plant_name required in JSON body',
                'example': {'plant_name': 'Rose', 'save_to_database': True},
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }), 400
        
        plant_name = data['plant_name'].strip()
        if not plant_name:
            return jsonify({
                'error': 'plant_name cannot be empty',
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }), 400
        
        # Generate plant data from APIs
        start_time = time.time()
        plant_data = kg.generate_plant_data_from_web(plant_name)
        generation_time = time.time() - start_time
        
        # Determine which APIs were used (simplified logic)
        apis_used = []
        if plant_data.get('scientific_name') != 'Unknown':
            apis_used.append('Wikipedia')
        if plant_data.get('family') != 'Unknown Family':
            apis_used.append('GBIF')
        if not apis_used:
            apis_used = ['Fallback Generator']
        
        # Optionally save to database
        save_to_db = data.get('save_to_database', True)
        database_result = {'success': False, 'message': 'Not attempted'}
        
        if save_to_db:
            save_start = time.time()
            cypher_query = kg.template_enhanced_cypher_insert(plant_data)
            success, message = kg.insert_plant_data(cypher_query)
            save_time = time.time() - save_start
            
            database_result = {
                'success': success,
                'message': message,
                'save_time': f"{save_time:.3f}s"
            }
        
        return jsonify({
            'success': True,
            'plant_name': plant_name,
            'generated_data': plant_data,
            'generation_time': f"{generation_time:.3f}s",
            'apis_used': apis_used,
            'database_save': database_result,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Data generation failed: {str(e)}",
            'error_type': type(e).__name__,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }), 500

# =============================================================================
# ADDITIONAL UTILITY ENDPOINTS
# =============================================================================

@app.route('/health')
def health_check():
    """Basic health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'services': {
            'flask_app': True,
            'neo4j_connection': bool(kg.driver),
            'model': os.path.exists(KERAS_MODEL_PATH),
            'class_indices': os.path.exists(CLASS_INDICES_PATH)
        }
    })

@app.route('/endpoints')
def list_endpoints():
    """List all available endpoints with descriptions"""
    endpoints = {
        'core': {
            'GET /': 'Home page with API information',
            'GET /status': 'Enhanced system status',
            'GET /health': 'Basic health check'
        },
        'testing': {
            'GET /test_connection': 'Test Neo4j database connection',
            'GET /test_apis': 'Test external API connectivity',
        },
        'information': {
            'GET /api_status': 'API sources status and capabilities',
            'GET /endpoints': 'List all available endpoints'
        },
        'search': {
            'GET /search/<plant_name>': 'Search existing knowledge graph data',
            'GET /smart_search/<plant_name>': 'Smart search with API auto-generation'
        },
        'prediction': {
            'GET /predict': 'Prediction endpoint information',
            'POST /predict': 'Upload image for plant species identification'
        },
        'data_generation': {
            'POST /generate_plant_data': 'Manual plant data generation from APIs'
        }
    }
    
    return jsonify({
        'service': 'Enhanced Plant Knowledge Graph API',
        'version': '2.0.0',
        'total_endpoints': sum(len(category) for category in endpoints.values()),
        'endpoints': endpoints,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    })

@app.errorhandler(404)
def not_found(error):
    """Custom 404 error handler"""
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': '/endpoints',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Custom 500 error handler"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred',
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }), 500

# =============================================================================
# APPLICATION STARTUP
# =============================================================================

if __name__ == "__main__":
    print("🌿 Starting Enhanced Plant Knowledge Graph Flask App...")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Port: {PORT}")
    print("\n🌐 API Data Sources:")
    print("- Wikipedia API for plant summaries")
    print("- GBIF (Global Biodiversity Information Facility)")
    print("- Tropicos (Missouri Botanical Garden)")
    print("- Smart search with fallback generation")
    
    print("CORE:")
    print("- GET  /                          - Home/API info")
    print("- GET  /status                    - System status")
    print("- GET  /health                    - Health check")
    print("- GET  /endpoints                 - List all endpoints")
    
    print("TESTING:")
    print("- GET  /test_connection           - Test DB connection")
    print("- GET  /test_apis                 - Test external APIs")
    
    print("INFORMATION:")
    print("- GET  /api_status                - API sources status")
    
    print("SEARCH:")
    print("- GET  /search/<plant_name>       - Search existing plants")
    print("- GET  /smart_search/<plant_name> - Smart search with API generation")
    
    print("PREDICTION:")
    print("- GET  /predict                   - Prediction info")
    print("- POST /predict                   - Image plant identification")
    
    print("DATA GENERATION:")
    print("- POST /generate_plant_data       - Manual data generation from APIs")
    
    print(f"\n🚀 Starting server on http://0.0.0.0:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)