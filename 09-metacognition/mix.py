import os
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from datetime import datetime
import requests
from openai import AzureOpenAI
from bs4 import BeautifulSoup
import re
import time
import random
import math
import sys
import subprocess
from duckduckgo_search import DDGS

# Check if required packages are installed
def check_and_install_packages():
    required_packages = ['requests', 'bs4', 'matplotlib', 'duckduckgo_search']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing required package: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"Successfully installed {package}")

# Run the package check at startup
check_and_install_packages()

# Initialize the Azure AI client
client = AzureOpenAI(
    api_key=os.environ["GITHUB_TOKEN"],  
    api_version="2023-05-15",
    azure_endpoint="https://models.inference.ai.azure.com",
)

# Create a SQLite database for our travel app
def setup_database():
    """Set up the SQLite database with sample travel data"""
    conn = sqlite3.connect('travel.db')
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS destinations (
        id INTEGER PRIMARY KEY,
        name TEXT,
        country TEXT,
        description TEXT,
        best_season TEXT,
        average_cost REAL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS hotels (
        id INTEGER PRIMARY KEY,
        name TEXT,
        destination_id INTEGER,
        price_per_night REAL,
        rating REAL,
        amenities TEXT,
        FOREIGN KEY (destination_id) REFERENCES destinations(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS attractions (
        id INTEGER PRIMARY KEY,
        name TEXT,
        destination_id INTEGER,
        category TEXT,
        rating REAL,
        price REAL,
        description TEXT,
        FOREIGN KEY (destination_id) REFERENCES destinations(id)
    )
    ''')
    
    # Sample data for destinations
    destinations = [
        (1, 'Paris', 'France', 'City of lights and romance', 'Spring', 200.0),
        (2, 'Tokyo', 'Japan', 'Blend of traditional and ultramodern', 'Spring', 250.0),
        (3, 'New York', 'USA', 'The city that never sleeps', 'Fall', 300.0),
        (4, 'Rome', 'Italy', 'The Eternal City', 'Spring', 180.0),
        (5, 'Sydney', 'Australia', 'Harbor city with iconic landmarks', 'Summer', 220.0)
    ]
    
    # Sample data for hotels
    hotels = [
        (1, 'Hotel de Paris', 1, 200.0, 4.5, 'WiFi, Breakfast, Pool'),
        (2, 'Luxe Parisienne', 1, 350.0, 4.8, 'WiFi, Breakfast, Spa, Pool'),
        (3, 'Budget Paris Inn', 1, 120.0, 3.5, 'WiFi'),
        (4, 'Tokyo Towers', 2, 280.0, 4.7, 'WiFi, Breakfast, Gym'),
        (5, 'Edo Heritage', 2, 220.0, 4.2, 'WiFi, Traditional Breakfast'),
        (6, 'Manhattan Suites', 3, 320.0, 4.6, 'WiFi, Breakfast, Gym, Business Center'),
        (7, 'Brooklyn Heights', 3, 180.0, 3.8, 'WiFi, Continental Breakfast'),
        (8, 'Roman Holiday', 4, 190.0, 4.0, 'WiFi, Breakfast, Terrace'),
        (9, 'Vatican View', 4, 280.0, 4.5, 'WiFi, Breakfast, Rooftop Restaurant'),
        (10, 'Sydney Harbor View', 5, 250.0, 4.4, 'WiFi, Breakfast, Pool, Ocean View'),
        (11, 'Bondi Beach Resort', 5, 210.0, 4.1, 'WiFi, Breakfast, Beach Access')
    ]
    
    # Sample data for attractions
    attractions = [
        (1, 'Eiffel Tower', 1, 'Landmark', 4.7, 25.0, 'Iconic iron tower with observation decks'),
        (2, 'Louvre Museum', 1, 'Museum', 4.8, 17.0, 'World-renowned art museum hosting ancient & medieval exhibits plus the iconic Mona Lisa'),
        (3, 'Notre-Dame Cathedral', 1, 'Religious', 4.6, 0.0, 'Medieval Catholic cathedral with flying buttresses & gargoyles'),
        (4, 'Tokyo Skytree', 2, 'Landmark', 4.5, 20.0, 'Tallest structure in Japan with observation decks'),
        (5, 'Senso-ji Temple', 2, 'Religious', 4.7, 0.0, 'Ancient Buddhist temple with a five-story pagoda'),
        (6, 'Tsukiji Fish Market', 2, 'Culinary', 4.4, 0.0, 'Bustling wholesale seafood market with sushi restaurants'),
        (7, 'Statue of Liberty', 3, 'Landmark', 4.7, 23.0, 'Iconic copper statue gifted by France'),
        (8, 'Central Park', 3, 'Nature', 4.8, 0.0, 'Sprawling park with lakes, walking paths & a zoo'),
        (9, 'Metropolitan Museum of Art', 3, 'Museum', 4.9, 25.0, 'Vast collection of art spanning 5,000+ years'),
        (10, 'Colosseum', 4, 'Historical', 4.8, 16.0, 'Iconic ancient Roman gladiatorial arena'),
        (11, 'Vatican Museums', 4, 'Museum', 4.7, 17.0, 'Museums featuring Michelangelo\'s Sistine Chapel'),
        (12, 'Trevi Fountain', 4, 'Landmark', 4.5, 0.0, 'Iconic 18th-century sculpted fountain'),
        (13, 'Sydney Opera House', 5, 'Landmark', 4.6, 40.0, 'Iconic performing arts center'),
        (14, 'Bondi Beach', 5, 'Nature', 4.7, 0.0, 'Popular beach known for surfing'),
        (15, 'Taronga Zoo', 5, 'Wildlife', 4.5, 44.0, 'Major zoo with Australian & exotic animals')
    ]
    
    # Insert data
    cursor.executemany('INSERT OR REPLACE INTO destinations VALUES (?,?,?,?,?,?)', destinations)
    cursor.executemany('INSERT OR REPLACE INTO hotels VALUES (?,?,?,?,?,?)', hotels)
    cursor.executemany('INSERT OR REPLACE INTO attractions VALUES (?,?,?,?,?,?,?)', attractions)
    
    conn.commit()
    conn.close()
    print("Database setup complete!")

class MetacognitiveAgent:
    """A metacognitive travel planning agent that can plan, reflect, and improve its recommendations"""
    
    def __init__(self, client):
        self.client = client
        self.user_preferences = {}
        self.experience_data = []
        self.reasoning_history = []
        self.current_strategy = "balanced"  # Initial strategy
        self.conn = sqlite3.connect('travel.db')
        
    def _log_reasoning(self, step, thought, outcome=None):
        """Log the agent's reasoning process for metacognition"""
        reasoning_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "thought": thought,
            "outcome": outcome
        }
        self.reasoning_history.append(reasoning_entry)
        print(f"🧠 {step}: {thought}")
        if outcome:
            print(f"📝 Outcome: {outcome}")
    
    def gather_preferences(self, preferences):
        """Gather user preferences for travel planning"""
        self.user_preferences = preferences
        self._log_reasoning(
            "Gather Preferences", 
            f"Collected user preferences for {preferences.get('destination', 'travel')}",
            f"Preferences stored: {json.dumps(preferences, indent=2)}"
        )
        
    def _generate_sql_query(self, table, filters=None, order_by=None, limit=None):
        """Generate a SQL query with filters, ordering, and limits"""
        query = f"SELECT * FROM {table}"
        
        if filters and len(filters) > 0:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, list):
                    placeholders = ', '.join(['?'] * len(value))
                    conditions.append(f"{key} IN ({placeholders})")
                else:
                    conditions.append(f"{key} = ?")
            
            query += " WHERE " + " AND ".join(conditions)
        
        if order_by:
            query += f" ORDER BY {order_by}"
            
        if limit:
            query += f" LIMIT {limit}"
            
        self._log_reasoning(
            "SQL Generation",
            f"Generated SQL query for {table}",
            f"Query: {query}"
        )
        
        return query
    
    def _execute_sql_query(self, query, params=None):
        """Execute a SQL query and return the results"""
        cursor = self.conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        columns = [description[0] for description in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            results.append(dict(zip(columns, row)))
            
        self._log_reasoning(
            "Database Query",
            f"Executed SQL query with {len(results)} results",
            f"First result (if any): {json.dumps(results[0] if results else {}, indent=2)}"
        )
        
        return results
    
    def _generate_code_for_data_retrieval(self, entity_type):
        """Generate Python code to retrieve data based on preferences"""
        code = f"""
def retrieve_{entity_type}_data(preferences):
    \"\"\"Retrieve {entity_type} data based on user preferences\"\"\"
    import sqlite3
    
    # Connect to the database
    conn = sqlite3.connect('travel.db')
    cursor = conn.cursor()
    
    # Build the query based on preferences
    query = "SELECT * FROM {entity_type}"
    conditions = []
    params = []
    
    if 'destination' in preferences:
        destination_query = "SELECT id FROM destinations WHERE name = ?"
        cursor.execute(destination_query, (preferences['destination'],))
        destination_id = cursor.fetchone()
        
        if destination_id:
            conditions.append("destination_id = ?")
            params.append(destination_id[0])
    
    if 'budget' in preferences:
        if preferences['budget'] == 'low':
            if '{entity_type}' == 'hotels':
                conditions.append("price_per_night < 150")
            elif '{entity_type}' == 'attractions':
                conditions.append("price < 20")
        elif preferences['budget'] == 'moderate':
            if '{entity_type}' == 'hotels':
                conditions.append("price_per_night BETWEEN 150 AND 250")
            elif '{entity_type}' == 'attractions':
                conditions.append("price BETWEEN 20 AND 40")
        elif preferences['budget'] == 'high':
            if '{entity_type}' == 'hotels':
                conditions.append("price_per_night > 250")
            elif '{entity_type}' == 'attractions':
                conditions.append("price > 40")
    
    if 'interests' in preferences and '{entity_type}' == 'attractions':
        placeholders = ', '.join(['?'] * len(preferences['interests']))
        conditions.append(f"category IN ({placeholders})")
        params.extend(preferences['interests'])
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    # Execute the query
    cursor.execute(query, params)
    
    # Get column names
    columns = [description[0] for description in cursor.description]
    
    # Fetch and format the results
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))
    
    conn.close()
    return results
"""
        self._log_reasoning(
            "Code Generation",
            f"Generated Python code for retrieving {entity_type} data",
            "Code generated successfully"
        )
        return code
    
    def _execute_generated_code(self, code, entity_type):
        """Execute the generated code to retrieve data"""
        try:
            # Create a local namespace for execution
            local_namespace = {"preferences": self.user_preferences}
            
            # Execute the code in the local namespace
            exec(code, globals(), local_namespace)
            
            # Get the function from the local namespace
            retrieve_function = local_namespace[f"retrieve_{entity_type}_data"]
            
            # Call the function with user preferences
            results = retrieve_function(self.user_preferences)
            
            self._log_reasoning(
                "Code Execution",
                f"Successfully executed code to retrieve {entity_type} data",
                f"Retrieved {len(results)} {entity_type}"
            )
            
            return results
        except Exception as e:
            self._log_reasoning(
                "Code Execution Error",
                f"Error executing code for {entity_type}: {str(e)}",
                "Failed to retrieve data"
            )
            return []
    
    def _get_destination_id(self, destination_name):
        """Get the destination ID from the name"""
        query = "SELECT id FROM destinations WHERE name = ?"
        cursor = self.conn.cursor()
        cursor.execute(query, (destination_name,))
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _apply_corrective_rag(self, initial_results, feedback):
        """Apply corrective RAG based on feedback to improve results"""
        self._log_reasoning(
            "Corrective RAG",
            "Applying feedback to improve results using RAG techniques",
            f"Feedback: {json.dumps(feedback, indent=2)}"
        )
        
        liked_items = feedback.get("liked", [])
        disliked_items = feedback.get("disliked", [])
        
        # Use the LLM to generate improved criteria based on feedback
        system_prompt = "You are a travel expert assistant that helps refine search criteria based on feedback."
        user_prompt = f"""
        I have the following preferences for my trip:
        {json.dumps(self.user_preferences, indent=2)}
        
        I liked these items: {', '.join(liked_items) if liked_items else 'None'}
        I disliked these items: {', '.join(disliked_items) if disliked_items else 'None'}
        
        Based on this feedback, suggest specific modifications to my search criteria to improve future results.
        Return your response as a JSON object with updated preferences.
        """
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        # Generate improved results using RAG with the model
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        # Extract and parse the improved results
        improved_results_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content in the response
            response_text = improved_results_text
            
            # Try to find JSON content (often between triple backticks)
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                updated_preferences = json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                updated_preferences = json.loads(response_text)
                
            self._log_reasoning(
                "Preference Update",
                "Successfully updated preferences based on feedback",
                f"Updated preferences: {json.dumps(updated_preferences, indent=2)}"
            )
            
            # Update user preferences with new values
            for key, value in updated_preferences.items():
                self.user_preferences[key] = value
                
            return True
        except Exception as e:
            self._log_reasoning(
                "Preference Update Error",
                f"Failed to update preferences based on feedback: {str(e)}",
                "Using original preferences"
            )
            return False
    
    def _score_relevance(self, item, preferences):
        """Score the relevance of an item based on user preferences"""
        score = 0
        
        # Base score calculation logic
        if 'budget' in preferences:
            if preferences['budget'] == 'low':
                if 'price_per_night' in item and item['price_per_night'] < 150:
                    score += 2
                elif 'price' in item and item['price'] < 20:
                    score += 2
            elif preferences['budget'] == 'moderate':
                if 'price_per_night' in item and 150 <= item['price_per_night'] <= 250:
                    score += 2
                elif 'price' in item and 20 <= item['price'] <= 40:
                    score += 2
            elif preferences['budget'] == 'high':
                if 'price_per_night' in item and item['price_per_night'] > 250:
                    score += 2
                elif 'price' in item and item['price'] > 40:
                    score += 2
        
        # Rating-based scoring
        if 'rating' in item:
            score += min(item['rating'], 5)  # Add up to 5 points for rating
        
        # Interest-based scoring for attractions
        if 'interests' in preferences and 'category' in item:
            if item['category'] in preferences['interests']:
                score += 3
        
        # Score boost for liked items
        if 'favorites' in preferences:
            if 'name' in item and item['name'] in preferences['favorites']:
                score += 5
        
        # Score penalty for disliked items
        if 'avoid' in preferences:
            if 'name' in item and item['name'] in preferences['avoid']:
                score -= 10
                
        return score
    
    def _rank_results(self, items, preferences):
        """Rank items based on relevance score"""
        scored_items = [(item, self._score_relevance(item, preferences)) for item in items]
        ranked_items = sorted(scored_items, key=lambda x: x[1], reverse=True)
        
        self._log_reasoning(
            "Result Ranking",
            f"Ranked {len(items)} items based on relevance to user preferences",
            f"Top score: {ranked_items[0][1] if ranked_items else 'N/A'}"
        )
        
        return [item for item, score in ranked_items]
    
    def bootstrap_itinerary_plan(self):
        """Bootstrap an initial travel itinerary plan based on user preferences"""
        self._log_reasoning(
            "Bootstrapping Plan",
            "Creating initial travel plan based on user preferences",
            f"Destination: {self.user_preferences.get('destination', 'Unknown')}"
        )
        
        # Get destination information
        destination_query = "SELECT * FROM destinations WHERE name = ?"
        destination_data = self._execute_sql_query(destination_query, (self.user_preferences.get('destination'),))
        
        if not destination_data:
            self._log_reasoning(
                "Bootstrapping Error",
                f"Destination '{self.user_preferences.get('destination')}' not found in database",
                "Cannot proceed with planning"
            )
            return None
        
        # Get destination ID
        destination_id = destination_data[0]['id']
        
        # Get hotel options from database based on budget
        hotel_filters = {"destination_id": destination_id}
        if 'budget' in self.user_preferences:
            budget_level = self.user_preferences['budget']
            if budget_level == 'low':
                hotel_query = """
                SELECT * FROM hotels 
                WHERE destination_id = ? AND price_per_night < 150
                ORDER BY rating DESC
                LIMIT 3
                """
            elif budget_level == 'high':
                hotel_query = """
                SELECT * FROM hotels 
                WHERE destination_id = ? AND price_per_night > 250
                ORDER BY rating DESC
                LIMIT 3
                """
            else:  # moderate
                hotel_query = """
                SELECT * FROM hotels 
                WHERE destination_id = ? AND price_per_night BETWEEN 150 AND 250
                ORDER BY rating DESC
                LIMIT 3
                """
        else:
            hotel_query = """
            SELECT * FROM hotels 
            WHERE destination_id = ?
            ORDER BY rating DESC
            LIMIT 3
            """
        
        hotels = self._execute_sql_query(hotel_query, (destination_id,))
        
        # Also get hotels from web search to provide more options
        destination_name = self.user_preferences.get('destination')
        budget_level = self.user_preferences.get('budget', 'moderate')
        
        # Extract check-in and check-out dates if available
        dates = self.user_preferences.get('dates', '')
        check_in_date = None
        check_out_date = None
        
        if dates:
            date_parts = dates.split(' to ')
            if len(date_parts) == 2:
                check_in_date = date_parts[0]
                check_out_date = date_parts[1]
        
        # Perform web search for hotels
        web_hotels = self.web_search_hotels(
            destination=destination_name,
            check_in_date=check_in_date,
            check_out_date=check_out_date,
            budget=budget_level
        )
        
        # Combine hotels from database and web search
        all_hotels = hotels.copy()
        
        # Add web hotels to the list, avoiding duplicates by name
        db_hotel_names = {hotel['name'].lower() for hotel in hotels}
        for web_hotel in web_hotels:
            if web_hotel['name'].lower() not in db_hotel_names:
                # Add destination_id to match database schema
                web_hotel['destination_id'] = destination_id
                all_hotels.append(web_hotel)
        
        self._log_reasoning(
            "Hotel Search Combined",
            f"Combined hotels from database ({len(hotels)}) and web search ({len(web_hotels)})",
            f"Total unique hotels: {len(all_hotels)}"
        )
        
        # Get attraction options based on interests
        attraction_query_base = "SELECT * FROM attractions WHERE destination_id = ?"
        params = [destination_id]
        
        if 'interests' in self.user_preferences and self.user_preferences['interests']:
            placeholders = ', '.join(['?'] * len(self.user_preferences['interests']))
            attraction_query = f"{attraction_query_base} AND category IN ({placeholders}) ORDER BY rating DESC"
            params.extend(self.user_preferences['interests'])
        else:
            attraction_query = f"{attraction_query_base} ORDER BY rating DESC"
        
        attractions = self._execute_sql_query(attraction_query, params)
        
        # Create initial itinerary
        itinerary = {
            "destination": destination_data[0],
            "hotels": all_hotels[:3],  # Top 3 hotels
            "attractions": attractions[:5],  # Top 5 attractions
            "dates": self.user_preferences.get('dates', 'Not specified'),
            "budget": self.user_preferences.get('budget', 'Not specified')
        }
        
        self._log_reasoning(
            "Initial Itinerary",
            "Successfully created initial travel itinerary",
            f"Included {len(all_hotels)} hotels and {len(attractions)} attractions"
        )
        
        return itinerary
    
    def reflect_on_itinerary(self, itinerary):
        """Reflect on the quality and completeness of the itinerary"""
        self._log_reasoning(
            "Itinerary Reflection",
            "Evaluating the quality and completeness of the generated itinerary",
            f"Analyzing {len(itinerary.get('hotels', []))} hotels and {len(itinerary.get('attractions', []))} attractions"
        )
        
        # Use LLM to evaluate the itinerary
        messages = [
            {
                "role": "system", 
                "content": """You are a travel expert that evaluates travel itineraries for quality and completeness.
                              Identify any issues or improvements that could be made."""
            },
            {
                "role": "user", 
                "content": f"""
                Please evaluate this travel itinerary for quality and completeness:
                
                Destination: {itinerary['destination']['name']}, {itinerary['destination']['country']}
                Travel Dates: {itinerary.get('dates', 'Not specified')}
                Budget Level: {itinerary.get('budget', 'Not specified')}
                
                Selected Hotels:
                {json.dumps([h['name'] for h in itinerary.get('hotels', [])], indent=2)}
                
                Selected Attractions:
                {json.dumps([a['name'] for a in itinerary.get('attractions', [])], indent=2)}
                
                User Preferences: {json.dumps(self.user_preferences, indent=2)}
                
                Identify any issues with this itinerary and suggest specific improvements.
                Return your analysis with:
                1. A quality score from 1-10
                2. Specific issues identified
                3. Specific improvements that could be made
                4. Any additional categories of activities that should be included
                """
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        response_content = response.choices[0].message.content
        
        self._log_reasoning(
            "Itinerary Quality Assessment",
            "Completed assessment of itinerary quality",
            response_content
        )
        
        return response_content
    
    def improve_itinerary(self, itinerary, reflection):
        """Improve the itinerary based on reflection insights"""
        self._log_reasoning(
            "Itinerary Improvement",
            "Applying reflective insights to improve the itinerary",
            "Analyzing reflection and making adjustments"
        )
        
        # Use LLM to extract specific improvements from the reflection
        messages = [
            {
                "role": "system", 
                "content": """You are a travel planning assistant that helps improve itineraries.
                              Extract actionable improvements from a reflection and apply them."""
            },
            {
                "role": "user", 
                "content": f"""
                Based on this reflection on a travel itinerary:
                
                {reflection}
                
                Generate specific, actionable changes to improve the itinerary.
                Return your response as a JSON object with these fields:
                - additional_interests: list of new interest categories to include
                - hotel_preferences: specific criteria to look for in hotels
                - attraction_preferences: specific criteria to look for in attractions 
                - changes_to_budget: any recommended budget adjustments
                """
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        improved_itinerary_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content in the response
            response_text = improved_itinerary_text
            
            # Try to find JSON content (often between triple backticks)
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1).strip()
                improvements = json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                improvements = json.loads(response_text)
                
            self._log_reasoning(
                "Extracted Improvements",
                "Successfully extracted improvement suggestions",
                f"Improvements: {json.dumps(improvements, indent=2)}"
            )
            
            # Update preferences based on improvements
            if 'additional_interests' in improvements and improvements['additional_interests']:
                current_interests = self.user_preferences.get('interests', [])
                if not isinstance(current_interests, list):
                    current_interests = [current_interests]
                
                # Add new interests
                current_interests.extend([i for i in improvements['additional_interests'] if i not in current_interests])
                self.user_preferences['interests'] = current_interests
            
            # Update hotel preferences
            if 'hotel_preferences' in improvements:
                for key, value in improvements['hotel_preferences'].items():
                    self.user_preferences[f'hotel_{key}'] = value
            
            # Update attraction preferences  
            if 'attraction_preferences' in improvements:
                for key, value in improvements['attraction_preferences'].items():
                    self.user_preferences[f'attraction_{key}'] = value
            
            # Update budget if needed
            if 'changes_to_budget' in improvements and improvements['changes_to_budget']:
                self.user_preferences['budget'] = improvements['changes_to_budget']
            
            # Regenerate the itinerary with updated preferences
            updated_itinerary = self.bootstrap_itinerary_plan()
            
            self._log_reasoning(
                "Itinerary Improved",
                "Successfully improved itinerary based on reflection",
                f"Updated with {len(updated_itinerary.get('hotels', []))} hotels and {len(updated_itinerary.get('attractions', []))} attractions"
            )
            
            return updated_itinerary
        except Exception as e:
            self._log_reasoning(
                "Improvement Error",
                f"Failed to extract improvements from reflection: {str(e)}",
                "Keeping original itinerary"
            )
            return itinerary
    
    def generate_day_by_day_plan(self, itinerary, num_days):
        """Generate a day-by-day plan for the itinerary"""
        self._log_reasoning(
            "Day-by-Day Planning",
            f"Generating a {num_days}-day plan for the itinerary",
            f"Planning activities for {itinerary['destination']['name']}"
        )
        
        # Use LLM to create a day-by-day plan
        messages = [
            {
                "role": "system", 
                "content": """You are a travel planning assistant that creates detailed day-by-day itineraries.
                              Create an optimal plan that balances activities throughout the stay."""
            },
            {
                "role": "user", 
                "content": f"""
                Create a detailed day-by-day itinerary for a {num_days}-day trip to {itinerary['destination']['name']}.
                
                Available Hotels (pick one):
                {json.dumps([{"name": h['name'], "rating": h['rating'], "amenities": h.get('amenities', 'WiFi, Breakfast')} for h in itinerary.get('hotels', [])], indent=2)}
                
                Available Attractions:
                {json.dumps([{"name": a['name'], "category": a['category'], "price": a['price'], "description": a['description']} for a in itinerary.get('attractions', [])], indent=2)}
                
                User Preferences: {json.dumps(self.user_preferences, indent=2)}
                
                For each day, include:
                1. Morning activities
                2. Lunch recommendations
                3. Afternoon activities
                4. Dinner recommendations
                5. Evening activities (if applicable)
                
                Provide a coherent plan that makes geographical sense (group nearby attractions), account for opening hours, and include time for rest. Include at least one local culinary experience.
                """
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        
        day_by_day_plan = response.choices[0].message.content
        
        self._log_reasoning(
            "Day-by-Day Planning",
            "Generated detailed daily itinerary",
            day_by_day_plan[:100] + "..." if len(day_by_day_plan) > 100 else day_by_day_plan
        )
        
        return day_by_day_plan
    
    def visualize_itinerary(self, itinerary):
        """Create visualizations for the itinerary"""
        self._log_reasoning(
            "Visualization",
            "Creating visualizations for the itinerary",
            "Generating charts for attractions and hotels"
        )
        
        # Create DataFrame for attractions
        attraction_data = []
        for attraction in itinerary.get('attractions', []):
            attraction_data.append({
                'Name': attraction['name'],
                'Category': attraction['category'],
                'Rating': attraction['rating'],
                'Price': attraction['price'] if attraction['price'] else 0
            })
        
        attractions_df = pd.DataFrame(attraction_data)
        
        # Create DataFrame for hotels
        hotel_data = []
        for hotel in itinerary.get('hotels', []):
            hotel_data.append({
                'Name': hotel['name'],
                'Rating': hotel['rating'],
                'Price': hotel['price_per_night']
            })
        
        hotels_df = pd.DataFrame(hotel_data)
        
        # Plot 1: Attraction Ratings by Category
        plt.figure(figsize=(12, 6))
        ax = attractions_df.plot.bar(x='Name', y='Rating', color='skyblue', rot=45)
        plt.title(f'Attraction Ratings in {itinerary["destination"]["name"]}')
        plt.xlabel('Attraction')
        plt.ylabel('Rating (out of 5)')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('attraction_ratings.png')
        
        # Plot 2: Hotel Price vs Rating
        plt.figure(figsize=(12, 6))
        plt.scatter(hotels_df['Price'], hotels_df['Rating'], s=100, alpha=0.7)
        for i, row in hotels_df.iterrows():
            plt.annotate(row['Name'], (row['Price'], row['Rating']), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Hotel Price vs. Rating in {itinerary["destination"]["name"]}')
        plt.xlabel('Price per Night ($)')
        plt.ylabel('Rating (out of 5)')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('hotel_comparison.png')
        
        # Plot 3: Attraction Prices
        plt.figure(figsize=(12, 6))
        ax = attractions_df.plot.bar(x='Name', y='Price', color='lightgreen', rot=45)
        plt.title(f'Attraction Prices in {itinerary["destination"]["name"]}')
        plt.xlabel('Attraction')
        plt.ylabel('Price ($)')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('attraction_prices.png')
        
        self._log_reasoning(
            "Visualization Complete",
            "Created 3 visualizations for the itinerary",
            "Saved charts as PNG files"
        )
        
        return {
            'attraction_ratings': 'attraction_ratings.png',
            'hotel_comparison': 'hotel_comparison.png',
            'attraction_prices': 'attraction_prices.png'
        }
    
    def plan_trip(self, destination, dates, budget, interests):
        """Main method to plan a complete trip"""
        self._log_reasoning(
            "Trip Planning",
            f"Starting trip planning process for {destination}",
            f"Planning for dates: {dates}, budget: {budget}, interests: {interests}"
        )
        
        # Gather user preferences
        preferences = {
            "destination": destination,
            "dates": dates,
            "budget": budget,
            "interests": interests
        }
        
        self.gather_preferences(preferences)
        
        # Step 1: Bootstrap initial itinerary
        initial_itinerary = self.bootstrap_itinerary_plan()
        if not initial_itinerary:
            return "Failed to create itinerary. Destination may not be in our database."
        
        # Step 2: Reflect on the initial itinerary
        reflection = self.reflect_on_itinerary(initial_itinerary)
        
        # Step 3: Improve the itinerary based on reflection
        improved_itinerary = self.improve_itinerary(initial_itinerary, reflection)
        
        # Step 4: Create a day-by-day plan
        start_date, end_date = dates.split(" to ")
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = (end - start).days + 1
        
        day_by_day_plan = self.generate_day_by_day_plan(improved_itinerary, num_days)
        
        # Step 5: Create visualizations
        visualizations = self.visualize_itinerary(improved_itinerary)
        
        # Prepare the final result
        result = {
            "destination_info": improved_itinerary["destination"],
            "selected_hotels": improved_itinerary["hotels"],
            "selected_attractions": improved_itinerary["attractions"],
            "day_by_day_plan": day_by_day_plan,
            "visualizations": visualizations,
            "reasoning_history": self.reasoning_history
        }
        
        self._log_reasoning(
            "Trip Planning Complete",
            f"Successfully planned trip to {destination}",
            f"Generated {num_days}-day itinerary with {len(improved_itinerary['hotels'])} hotels and {len(improved_itinerary['attractions'])} attractions"
        )
        
        return result
    
    def web_search_hotels(self, destination, check_in_date=None, check_out_date=None, budget=None):
        """
        Search for hotels using DuckDuckGo search without requiring API keys.
        
        Args:
            destination (str): The destination city/location
            check_in_date (str, optional): Check-in date in YYYY-MM-DD format
            check_out_date (str, optional): Check-out date in YYYY-MM-DD format
            budget (str, optional): Budget category ('low', 'moderate', 'high')
            
        Returns:
            list: A list of hotel dictionaries with information
        """
        self._log_reasoning(
            "Web Hotel Search",
            f"Searching for hotels in {destination} using DuckDuckGo",
            f"Parameters: check_in={check_in_date}, check_out={check_out_date}, budget={budget}"
        )
        
        try:
            # Prepare search terms
            search_query = f"best hotels in {destination}"
            if budget:
                if budget == "low":
                    search_query += " budget affordable"
                elif budget == "moderate":
                    search_query += " mid-range"
                elif budget == "high":
                    search_query += " luxury"
            
            if check_in_date and check_out_date:
                search_query += f" {check_in_date} to {check_out_date}"
            
            # Initialize the DuckDuckGo search client
            ddgs = DDGS()
            
            # Perform the search (max 25 results)
            results = list(ddgs.text(search_query, max_results=25))
            
            if results:
                # Process the search results into hotel data
                hotels = []
                
                for i, result in enumerate(results):
                    # Skip results that are clearly not hotels
                    title = result.get('title', '')
                    body = result.get('body', '')
                    href = result.get('href', '')
                    
                    # Simple heuristic to filter out non-hotel results
                    hotel_keywords = ['hotel', 'resort', 'inn', 'suites', 'lodging', 'accommodation']
                    booking_sites = ['booking.com', 'expedia', 'hotels.com', 'tripadvisor', 'airbnb']
                    
                    is_likely_hotel = (
                        any(keyword.lower() in title.lower() for keyword in hotel_keywords) or
                        any(site in href.lower() for site in booking_sites)
                    )
                    
                    if not is_likely_hotel:
                        continue
                    
                    # Extract rating if available
                    rating_match = re.search(r'(\d\.\d+)/5', body) or re.search(r'(\d\.\d+) out of 5', body) or re.search(r'(\d\.\d+) stars', body)
                    rating = float(rating_match.group(1)) if rating_match else random.uniform(3.5, 4.8)
                    
                    # Extract price if available
                    price_match = re.search(r'\$(\d+)', body) or re.search(r'(\d+) per night', body)
                    if price_match:
                        price = float(price_match.group(1))
                    else:
                        # Assign price based on budget
                        if budget == "low":
                            price = random.uniform(80, 150)
                        elif budget == "moderate":
                            price = random.uniform(150, 300)
                        elif budget == "high":
                            price = random.uniform(300, 800)
                        else:
                            price = random.uniform(100, 400)
                    
                    # Prepare title
                    title_cleaned = re.sub(r' - .*$', '', title)  # Remove everything after dash
                    title_cleaned = re.sub(r' \|.*$', '', title_cleaned)  # Remove everything after pipe
                    
                    # Generate random amenities based on hotel quality/price
                    amenities_options = [
                        "WiFi, Breakfast, Air conditioning",
                        "WiFi, Room service, TV",
                        "WiFi, Pool, Fitness center",
                        "WiFi, Restaurant, Bar",
                        "WiFi, Spa, Concierge"
                    ]
                    
                    # More expensive hotels get better amenities
                    amenities_index = min(int(rating - 3), 4) if rating > 3 else 0
                    amenities = amenities_options[amenities_index]
                    
                    # Create hotel object
                    hotel = {
                        "id": i + 1000,  # Use high IDs to avoid conflicts with database
                        "name": title_cleaned,
                        "destination": destination,
                        "price_per_night": round(price, 2),
                        "rating": round(rating, 1),
                        "description": body,
                        "source_url": href,
                        "amenities": amenities,
                        "from_web_search": True
                    }
                    
                    hotels.append(hotel)
                
                self._log_reasoning(
                    "Web Hotel Search Results",
                    f"Found {len(hotels)} hotels for {destination} using DuckDuckGo",
                    f"Sample result: {json.dumps(hotels[0] if hotels else {}, indent=2)}"
                )
                
                return hotels
            else:
                self._log_reasoning(
                    "Web Hotel Search Notice",
                    f"No results found through DuckDuckGo for {destination}",
                    "Using backup approach with simulated results"
                )
                return self._generate_simulated_hotel_results(destination, budget)
                
        except Exception as e:
            self._log_reasoning(
                "Web Hotel Search Error",
                f"Error using DuckDuckGo search: {str(e)}",
                "Using backup approach with simulated results"
            )
            return self._generate_simulated_hotel_results(destination, budget)
    
    def _generate_simulated_hotel_results(self, destination, budget=None):
        """Generate simulated hotel results when web scraping fails"""
        hotels = []
        
        # Hotel name templates based on destination
        hotel_prefixes = ["Grand Hotel", "Royal", "Plaza", "Majestic", "Luxe", "Comfort Inn", "Urban Stay", "Riverside"]
        hotel_suffixes = ["Resort & Spa", "Palace", "Suites", "Inn", "Hotel", "Lodging", "BnB", "Apartments"]
        
        # Amenities options
        amenities_options = [
            "WiFi, Pool, Spa, Restaurant",
            "WiFi, Breakfast, Fitness Center",
            "WiFi, Room Service, Bar",
            "Breakfast, Parking, WiFi",
            "WiFi, Laundry, Kitchen",
            "WiFi, Parking, Pet-friendly"
        ]
        
        # Generate 5-8 hotels
        num_hotels = random.randint(5, 8)
        
        for i in range(num_hotels):
            # Generate name
            prefix = random.choice(hotel_prefixes)
            suffix = random.choice(hotel_suffixes)
            name = f"{prefix} {destination} {suffix}"
            
            # Generate rating (3.0-5.0)
            rating = round(random.uniform(3.0, 5.0), 1)
            
            # Generate price based on budget and rating
            base_price = rating * 30  # Higher rating = higher base price
            
            if budget == "low":
                price_factor = random.uniform(0.7, 1.0)
                price = round(base_price * price_factor, 2)
            elif budget == "moderate":
                price_factor = random.uniform(1.0, 2.0)
                price = round(base_price * price_factor, 2)
            elif budget == "high":
                price_factor = random.uniform(2.0, 4.0)
                price = round(base_price * price_factor, 2)
            else:
                price_factor = random.uniform(0.8, 3.0)
                price = round(base_price * price_factor, 2)
            
            # Select amenities
            amenities = random.choice(amenities_options)
            
            # Generate description
            description = f"Located in the heart of {destination}, this {rating}-star hotel offers comfortable accommodation with {amenities.lower()}."
            
            # Create hotel object
            hotel = {
                "id": i + 2000,  # Use even higher IDs to distinguish simulated results
                "name": name,
                "destination": destination,
                "price_per_night": price,
                "rating": rating,
                "amenities": amenities,
                "description": description,
                "source": "simulated",
                "from_web_search": True
            }
            
            hotels.append(hotel)
        
        self._log_reasoning(
            "Simulated Hotel Results",
            f"Generated {len(hotels)} simulated hotel results for {destination}",
            f"Sample result: {json.dumps(hotels[0] if hotels else {}, indent=2)}"
        )
        
        return hotels

# Main execution
if __name__ == "__main__":
    # Set up the database
    setup_database()
    
    # Create the metacognitive agent
    agent = MetacognitiveAgent(client)
    
    # Demonstrate web hotel search functionality
    print("\n==== DEMONSTRATING DUCKDUCKGO HOTEL SEARCH ====")
    print("Searching for hotels in Barcelona using DuckDuckGo (no API key required)...")
    barcelona_hotels = agent.web_search_hotels(
        destination="Barcelona",
        budget="moderate"
    )
    
    print(f"\nFound {len(barcelona_hotels)} hotels in Barcelona via DuckDuckGo:")
    for i, hotel in enumerate(barcelona_hotels[:5]):  # Show up to 5 hotels
        print(f"{i+1}. {hotel['name']}")
        print(f"   Rating: {hotel['rating']}")
        print(f"   Price: ${hotel['price_per_night']}/night")
        print(f"   Description: {hotel['description'][:100]}..." if len(hotel['description']) > 100 else f"   Description: {hotel['description']}")
        print()
    
    # Plan a trip to Paris
    trip_plan = agent.plan_trip(
        destination="Paris",
        dates="2025-04-01 to 2025-04-07",
        budget="moderate",
        interests=["Museum", "Landmark", "Culinary"]
    )
    
    # Print the day-by-day plan
    print("\n==== DAY-BY-DAY PLAN ====")
    print(trip_plan["day_by_day_plan"])
    
    # Count hotels that came from web search
    web_hotels_count = sum(1 for hotel in trip_plan["selected_hotels"] if hotel.get('from_web_search', False))
    print(f"\n==== SELECTED HOTELS ({web_hotels_count} from DuckDuckGo search) ====")
    for hotel in trip_plan["selected_hotels"]:
        source = "(DuckDuckGo)" if hotel.get('from_web_search', False) else "(Database)"
        print(f"- {hotel['name']} {source} (Rating: {hotel['rating']}, Price: ${hotel['price_per_night']}/night)")
    
    print("\n==== SELECTED ATTRACTIONS ====")
    for attraction in trip_plan["selected_attractions"]:
        print(f"- {attraction['name']} ({attraction['category']}, Rating: {attraction['rating']})")
    
    print("\n==== VISUALIZATIONS ====")
    print(f"Visualizations saved to: {', '.join(trip_plan['visualizations'].values())}")
    
    # Apply corrective RAG with simulated feedback
    print("\n==== APPLYING FEEDBACK WITH CORRECTIVE RAG ====")
    feedback = {
        "liked": ["Louvre Museum", "Hotel de Paris"],
        "disliked": ["Eiffel Tower (too crowded)"]
    }
    
    agent._apply_corrective_rag(trip_plan["selected_attractions"], feedback)
    
    # Plan the trip again with updated preferences
    print("\n==== UPDATED TRIP PLAN AFTER FEEDBACK ====")
    updated_trip_plan = agent.plan_trip(
        destination="Paris",
        dates="2025-04-01 to 2025-04-07",
        budget="moderate",
        interests=["Museum", "Landmark", "Culinary"]
    )
    
    # Print the day-by-day plan
    print("\n==== DAY-BY-DAY PLAN ====")
    print(updated_trip_plan["day_by_day_plan"])
    
    # Count hotels that came from web search in updated trip
    updated_web_hotels_count = sum(1 for hotel in updated_trip_plan["selected_hotels"] if hotel.get('from_web_search', False))
    print(f"\n==== SELECTED HOTELS ({updated_web_hotels_count} from DuckDuckGo search) ====")
    for hotel in updated_trip_plan["selected_hotels"]:
        source = "(DuckDuckGo)" if hotel.get('from_web_search', False) else "(Database)"
        print(f"- {hotel['name']} {source} (Rating: {hotel['rating']}, Price: ${hotel['price_per_night']}/night)")
    
    print("\n==== SELECTED ATTRACTIONS ====")
    for attraction in updated_trip_plan["selected_attractions"]:
        print(f"- {attraction['name']} ({attraction['category']}, Rating: {attraction['rating']})")
    
    print("\n==== VISUALIZATIONS ====")
    print(f"Visualizations saved to: {', '.join(updated_trip_plan['visualizations'].values())}")
