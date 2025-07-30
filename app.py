from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import re
import warnings
from typing import Dict, List, Tuple
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

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
            
            # Initialize models for plant identification and data generation
        self.text_generator = None
            
            # Try to establish connection with multiple URI formats
        self._establish_connection()
    
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
            "🤖 AI-powered data generation",
            "🌐 Web scraping for plant data",
            "📊 Knowledge graph relationships"
        ],
        "endpoints": {
            "test_connection": "/test_connection",
            "search": "/search/<plant_name>",
            "smart_search": "/smart_search/<plant_name>",
            
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
                'text_generator': bool(kg.text_generator)
            }
        })
        
    except Exception as e:
        return jsonify({
            'demo_completed': False,
            'error': str(e)
        }), 500


    """Check AI models status"""
    return jsonify({
        'ai_models': {
          
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
            '🌐 Web scraping from botanical APIs',
            '📊 Relationship mapping in knowledge graph'
        ]
    })

if __name__ == "__main__":
    print("🌿 Starting Enhanced Plant Knowledge Graph Flask App...")
    print(f"Neo4j URI: {NEO4J_URI}")
    print(f"Port: {PORT}")
    print("\n🤖 AI Features:")
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
    print("- POST /generate_plant_data       - Manual data generation")
    print("- GET  /quick_demo                - Enhanced demo with AI features")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)