from . import consumer

channel_routing = {
    # TODO: From the original examples, there is more (https://github.com/jacobian/channels-example/)
    'websocket.connect': consumer.ws_connect,
    'websocket.receive': consumer.ws_receive,
    'websocket.disconnect': consumer.ws_disconnect,
}
