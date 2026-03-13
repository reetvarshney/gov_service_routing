"""
Portal Routing Module
Maps intents to government portals based on location
"""

import pandas as pd
import os

class PortalRouter:
    def __init__(self, portal_file="data/portals.csv"):
        """
        Initialize the router with portal mapping data
        
        Args:
            portal_file: Path to portal mapping CSV
        """
        self.portal_file = portal_file
        self.portals_df = None
        self.load_portals()
    
    def load_portals(self):
        """
        Load portal mapping from CSV file
        """
        if not os.path.exists(self.portal_file):
            print(f"Warning: Portal file not found at {self.portal_file}")
            # Create empty dataframe with required columns
            self.portals_df = pd.DataFrame(columns=['intent', 'state', 'department', 'portal_link'])
            return
        
        self.portals_df = pd.read_csv(self.portal_file)
        print(f"Loaded {len(self.portals_df)} portal mappings")
    
    def get_portal(self, intent, state="Uttarakhand"):
        """
        Get portal information for intent and state
        
        Args:
            intent: Predicted intent category
            state: User's state (default: Uttarakhand)
            
        Returns:
            Dictionary with portal info or None if not found
        """
        if self.portals_df is None or len(self.portals_df) == 0:
            return {
                'department': 'General Services Portal',
                'link': 'https://www.india.gov.in',
                'state': 'India',
                'note': 'Using generic portal (mapping not available)'
            }
        
        # Try to find state-specific portal
        result = self.portals_df[
            (self.portals_df['intent'].str.lower() == intent.lower()) & 
            (self.portals_df['state'].str.lower() == state.lower())
        ]
        
        if len(result) > 0:
            return {
                'department': result.iloc[0]['department'],
                'link': result.iloc[0]['portal_link'],
                'state': state,
                'note': 'State-specific portal'
            }
        
        # Try case-insensitive match for state
        result = self.portals_df[
            (self.portals_df['intent'].str.lower() == intent.lower()) & 
            (self.portals_df['state'].str.lower().str.contains(state.lower()))
        ]
        
        if len(result) > 0:
            return {
                'department': result.iloc[0]['department'],
                'link': result.iloc[0]['portal_link'],
                'state': result.iloc[0]['state'],
                'note': 'Similar state match'
            }
        
        # Fallback to any portal for this intent
        result = self.portals_df[self.portals_df['intent'].str.lower() == intent.lower()]
        
        if len(result) > 0:
            return {
                'department': result.iloc[0]['department'],
                'link': result.iloc[0]['portal_link'],
                'state': result.iloc[0]['state'],
                'note': 'Using portal from different state'
            }
        
        # Ultimate fallback
        result = self.portals_df[self.portals_df['intent'].str.lower() == 'other']
        if len(result) > 0:
            return {
                'department': result.iloc[0]['department'],
                'link': result.iloc[0]['portal_link'],
                'state': 'India',
                'note': 'Generic central portal'
            }
        
        # No match found
        return {
            'department': 'Government Services Portal',
            'link': 'https://www.india.gov.in',
            'state': 'India',
            'note': 'Please visit central portal'
        }
    
    def get_all_states(self):
        """
        Get list of all states in the mapping
        
        Returns:
            List of states
        """
        if self.portals_df is not None and len(self.portals_df) > 0:
            return sorted(self.portals_df['state'].unique().tolist())
        return ["Uttarakhand", "Uttar Pradesh", "Delhi", "Other"]
    
    def get_intents_for_state(self, state):
        """
        Get all intents available for a state
        
        Args:
            state: State name
            
        Returns:
            List of intents
        """
        if self.portals_df is None:
            return []
        
        state_portals = self.portals_df[self.portals_df['state'].str.lower() == state.lower()]
        return state_portals['intent'].unique().tolist()
    
    def add_portal(self, intent, state, department, link):
        """
        Add a new portal mapping (for future expansion)
        
        Args:
            intent: Intent category
            state: State name
            department: Department name
            link: Portal URL
        """
        new_row = pd.DataFrame({
            'intent': [intent],
            'state': [state],
            'department': [department],
            'portal_link': [link]
        })
        
        if self.portals_df is None:
            self.portals_df = new_row
        else:
            self.portals_df = pd.concat([self.portals_df, new_row], ignore_index=True)
        
        # Save to file
        self.portals_df.to_csv(self.portal_file, index=False)
        print(f"Added portal: {intent} - {state}")

# For testing
if __name__ == "__main__":
    router = PortalRouter()
    
    # Test with different intents and states
    test_cases = [
        ("Electricity", "Uttarakhand"),
        ("Water", "Delhi"),
        ("Healthcare", "Maharashtra"),
        ("Road", "Unknown State")
    ]
    
    for intent, state in test_cases:
        portal = router.get_portal(intent, state)
        print(f"\nIntent: {intent}, State: {state}")
        print(f"Department: {portal['department']}")
        print(f"Link: {portal['link']}")
        print(f"Note: {portal['note']}")