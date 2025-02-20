from django.http import JsonResponse
import json
from django.views.decorators.csrf import csrf_exempt
from django.views import View

class OrderTrackingHandler:
    _instance = None
    clients = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OrderTrackingHandler, cls).__new__(cls)
        return cls._instance

    def register_client(self, order_id, client):
        if order_id not in self.clients:
            self.clients[order_id] = set()
        self.clients[order_id].add(client)

    def unregister_client(self, order_id, client):
        if order_id in self.clients and client in self.clients[order_id]:
            self.clients[order_id].remove(client)
            if not self.clients[order_id]:
                del self.clients[order_id]

    def notify_status_update(self, order_id, status):
        if order_id in self.clients:
            message = json.dumps({
                'type': 'status_update',
                'status': status
            })
            for client in list(self.clients[order_id]):
                try:
                    client.send(message)
                except:
                    self.unregister_client(order_id, client)

    @csrf_exempt
    def handle_request(self, request, order_id):
        if request.method == 'GET':
            return JsonResponse({
                'type': 'connection_established',
                'order_id': order_id
            })
        return JsonResponse({'error': 'Method not allowed'}, status=405)

# Create a single instance
tracking_handler = OrderTrackingHandler()
