import { useState, useEffect, useRef, useCallback } from 'react';

function App() {
    const [status, setStatus] = useState('Connecting...');
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isConnected, setIsConnected] = useState(false);

    const pc = useRef(null);
    const dc = useRef(null);
    // This ref will hold the ID of the bot message currently being streamed.
    const streamingMessageId = useRef(null);

    const addMessage = useCallback((text, sender) => {
        const newMessage = { id: Date.now() + Math.random(), text, sender };
        setMessages(prev => [...prev, newMessage]);
        return newMessage.id;
    }, []);

    useEffect(() => {
        const config = { sdpSemantics: 'unified-plan' };
        pc.current = new RTCPeerConnection(config);

        const dataChannel = pc.current.createDataChannel('chat', { ordered: true });
        dc.current = dataChannel;

        dataChannel.onopen = () => {
            setStatus('Connected');
            setIsConnected(true);
        };

        dataChannel.onclose = () => {
            setStatus('Connection closed.');
            setIsConnected(false);
        };

        dataChannel.onmessage = (evt) => {
            // The backend now sends one complete message per event, ending with a newline.
            const messageText = evt.data.trim(); // Trim the newline
            if (messageText) {
                addMessage(messageText, 'bot');
            }
        };

        pc.current.oniceconnectionstatechange = () => {
            if (pc.current.iceConnectionState === 'failed') {
                setStatus('Connection failed.');
            }
        };

        negotiate();

        return () => {
            if (dataChannel) dataChannel.close();
            if (pc.current) pc.current.close();
        };
    }, []);

    const negotiate = async () => {
        try {
            const offer = await pc.current.createOffer();
            await pc.current.setLocalDescription(offer);
            
            await new Promise(resolve => {
                if (pc.current.iceGatheringState === 'complete') resolve();
                else pc.current.addEventListener('icegatheringstatechange', () => {
                    if (pc.current.iceGatheringState === 'complete') resolve();
                });
            });

            setStatus('Sending offer...');
            const response = await fetch('http://localhost:8080/offer', {
                body: JSON.stringify({ sdp: pc.current.localDescription.sdp, type: pc.current.localDescription.type }),
                headers: { 'Content-Type': 'application/json' },
                method: 'POST'
            });

            const answer = await response.json();
            setStatus('Verifying connection...');
            await pc.current.setRemoteDescription(answer);
        } catch (e) {
            setStatus(`Error: ${e.toString()}`);
            alert(`An error occurred during connection setup: ${e}`);
        }
    };

    const sendMessage = () => {
        if (inputValue.trim() === '' || !isConnected) return;

        addMessage(inputValue, 'user');
        dc.current.send(inputValue);
        setInputValue('');
        streamingMessageId.current = null; // User message interrupts any bot stream
    };

    return (
        <div className="chat-container">
            <header className="status-header">{status}</header>
            <main className="chat-log">
                {messages.map(msg => (
                    <div key={msg.id} className={`message ${msg.sender}-message`}>
                        {msg.text}
                    </div>
                ))}
            </main>
            <footer className="chat-input-container">
                <input
                    type="text"
                    className="message-input"
                    placeholder="Type your message..."
                    value={inputValue}
                    onChange={e => setInputValue(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && sendMessage()}
                    disabled={!isConnected}
                />
                <button
                    className="send-button"
                    onClick={sendMessage}
                    disabled={!isConnected}
                >
                    Send
                </button>
            </footer>
        </div>
    );
}

export default App;
