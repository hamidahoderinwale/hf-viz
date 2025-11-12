"""
Netlify serverless function wrapper for models API.
This file is the entry point for Netlify Functions.
"""
import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api import handler

# Netlify Functions expects a handler function
def lambda_handler(event, context):
    """
    AWS Lambda/Netlify Functions handler.
    Converts Netlify event format to our handler format.
    """
    # Convert Netlify event to our format
    # Netlify passes path in event['path']
    # Query params are in event['queryStringParameters']
    return handler(event, context)

