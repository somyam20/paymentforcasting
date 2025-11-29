# utils/obs.py

import os
import json
import time
import uuid
import jwt
import logging
from typing import Any, Dict, List, Optional
from fastapi import Request

# Import the singleton logger instance from your kafka module
from .kafka import kafka_logger

# Define the separator for parsing the auth token
CUSTOM_TOKEN_SEPARATOR = "$YashUnified2025$"

class LLMUsageTracker:
    """
    Unified usage tracker for Gemini SDK and dict responses. It processes usage data,
    enriches it with context, sends a specific, lean payload to Kafka,
    and returns a summary dictionary.
    """

    def __init__(self):
        # --- Authentication and Environment Configuration ---
        self.auth_token = None
        
        # --- Constants for the Kafka payload, sourced from environment variables ---
        self.agent_name = os.getenv("AGENT_NAME_CONSTANT", "Unknown Agent")
        self.server_name = os.getenv("SERVER_NAME", "Unknown Server")
        self.model_name_env = os.getenv("MODEL_NAME", "N/A")

    def track_response(self, response: Any, auth_token: str, model: str) -> Dict[str, Any]:
        """
        Main entry point: processes a response, logs the required details to Kafka,
        and returns a status dictionary.
        """
        try:
            self.auth_token = auth_token
            record = self._normalize_response(response)
            if not record:
                message = "Unrecognized response format; cannot process for Kafka logging."
                logging.error(message)
                return {"status": "error", "message": message}

            kafka_payload = self._prepare_kafka_payload(record)
            print("kafka_payload", kafka_payload)
            if kafka_payload.get("total_tokens", 0) > 0:
                kafka_logger.log(kafka_payload)
            else:
                logging.warning("Response record contains no token usage, skipping Kafka log.")

            return {
                "status": "success",
                "message": "Usage data has been queued for logging.",
                "details": {
                    "user": kafka_payload.get("user_email"),
                    "total_tokens": kafka_payload.get("total_tokens"),
                }
            }
        except Exception as e:
            logging.error(f"Error in track_response: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _normalize_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """Converts LiteLLM, Gemini SDK or dict response into a standardized dictionary."""
        
        # Handle LiteLLM response objects
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            normalized_usage = {
                "promptTokenCount": getattr(usage, "prompt_tokens", 0) or 0,
                "candidatesTokenCount": getattr(usage, "completion_tokens", 0) or 0,
                "totalTokenCount": getattr(usage, "total_tokens", 0) or 0,
            }
            return {"usageMetadata": normalized_usage}
        
        # Handle LiteLLM response as dict format
        if isinstance(response, dict) and "usage" in response:
            usage = response["usage"]
            normalized_usage = {
                "promptTokenCount": usage.get("prompt_tokens", 0) or 0,
                "candidatesTokenCount": usage.get("completion_tokens", 0) or 0,
                "totalTokenCount": usage.get("total_tokens", 0) or 0,
            }
            return {"usageMetadata": normalized_usage}
        
        # Handle legacy Gemini SDK format
        if hasattr(response, "usage_metadata"):
            um = response.usage_metadata
            usage = {
                "promptTokenCount": getattr(um, "prompt_token_count", 0) or 0,
                "candidatesTokenCount": getattr(um, "candidates_token_count", 0) or 0,
                "totalTokenCount": getattr(um, "total_token_count", 0) or 0,
            }
            return {"usageMetadata": usage}
        
        # Handle legacy dict format
        if isinstance(response, dict) and "usageMetadata" in response:
            return response
            
        return None

    def _prepare_kafka_payload(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts required metrics and builds the final dictionary for Kafka.
        """
        usage_meta = record.get("usageMetadata", {})
        
        prompt_tokens = usage_meta.get("promptTokenCount", 0) or 0
        completion_tokens = usage_meta.get("candidatesTokenCount", 0) or 0
        total_tokens = usage_meta.get("totalTokenCount", 0) or (prompt_tokens + completion_tokens)
        thoughts_tokens = usage_meta.get("thoughtsTokenCount", 0) or 0
        
        user_email, encrypted_payload = self._parse_auth_token()
        
        final_log = {
            "encrypted_payload": encrypted_payload,
            "user_email": user_email,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "thoughts_token_count": thoughts_tokens,
            "model_name": self.model_name_env,
            "agent_name_constant": self.agent_name,
            "server_name": self.server_name,
        }
        print(final_log)
        return final_log

    def _parse_auth_token(self) -> (str, str):
        """
        Safely parses the combined auth token string stored in the instance,
        handling the "Bearer" prefix.
        """
        if not self.auth_token:
            return "N/A", "N/A"
        
        token_to_process = self.auth_token
        encrypted_payload = "N/A"

        # Check for and split the custom encrypted payload part first
        if CUSTOM_TOKEN_SEPARATOR in token_to_process:
            parts = token_to_process.split(CUSTOM_TOKEN_SEPARATOR, 1)
            token_to_process = parts[0]
            encrypted_payload = parts[1] if len(parts) > 1 else "N/A"
        
        # ** FIX: Remove "Bearer " prefix if it exists **
        if token_to_process.lower().startswith("bearer "):
            jwt_part = token_to_process[7:]
        else:
            jwt_part = token_to_process
        
        try:
            # Now, decode the clean JWT part
            decoded_token = jwt.decode(jwt_part, options={"verify_signature": False})
            
            logging.info(f"Decoded JWT claims: {decoded_token}")
            
            custom_data = decoded_token.get("custom-data", {})
            user_email = custom_data.get("user_email") or decoded_token.get("email") or "N/A"
            
            logging.info(f"Extracted user_email: {user_email}")

        except jwt.PyJWTError as e:
            logging.warning(f"Could not decode JWT. Token part used: '{jwt_part[:30]}...'. Error: {e}")
            user_email = "N/A"
        
        return user_email, encrypted_payload