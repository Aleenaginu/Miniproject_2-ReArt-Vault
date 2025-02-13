from twilio.rest import Client

# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC09b6c386c220d7d6cbd24145c9f22b41'
TWILIO_AUTH_TOKEN = 'aa31693c5cec3f4cbfcacf1d414ea9fe'
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio's WhatsApp sandbox number
RECIPIENT_WHATSAPP_NUMBER = '+919188647088'  # Your WhatsApp number for testing

def send_whatsapp_message(to_number, message):
    """
    Send a WhatsApp message using Twilio
    :param to_number: The recipient's phone number (not used in sandbox mode)
    :param message: The message to send
    """
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        # In sandbox mode, we'll always send to the hardcoded number
        to_whatsapp_number = f"whatsapp:{RECIPIENT_WHATSAPP_NUMBER}"
        
        # Send message
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to_whatsapp_number
        )
        
        return True, message.sid
    except Exception as e:
        return False, str(e)
