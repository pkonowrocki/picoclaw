package channels

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/utils"
)

const (
	messengerGraphAPIBase    = "https://graph.facebook.com/v21.0"
	messengerMessagesAPI     = messengerGraphAPIBase + "/me/messages"
)

// MessengerChannel implements the Channel interface for Facebook Messenger
// using the Meta Webhook API for receiving messages and the Graph API for
// sending replies.
type MessengerChannel struct {
	*BaseChannel
	config     config.MessengerConfig
	httpServer *http.Server
	ctx        context.Context
	cancel     context.CancelFunc
}

// NewMessengerChannel creates a new Messenger channel instance.
func NewMessengerChannel(cfg config.MessengerConfig, messageBus *bus.MessageBus) (*MessengerChannel, error) {
	if cfg.PageAccessToken == "" || cfg.AppSecret == "" {
		return nil, fmt.Errorf("messenger page_access_token and app_secret are required")
	}

	base := NewBaseChannel("messenger", cfg, messageBus, cfg.AllowFrom)

	return &MessengerChannel{
		BaseChannel: base,
		config:      cfg,
	}, nil
}

// Start launches the HTTP webhook server for Messenger.
func (c *MessengerChannel) Start(ctx context.Context) error {
	logger.InfoC("messenger", "Starting Messenger channel (Webhook Mode)")

	c.ctx, c.cancel = context.WithCancel(ctx)

	mux := http.NewServeMux()
	path := c.config.WebhookPath
	if path == "" {
		path = "/messenger/webhook"
	}
	mux.HandleFunc(path, c.webhookHandler)

	port := c.config.WebhookPort
	if port == 0 {
		port = 18791
	}
	addr := fmt.Sprintf("0.0.0.0:%d", port)
	c.httpServer = &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	go func() {
		logger.InfoCF("messenger", "Messenger webhook server listening", map[string]interface{}{
			"addr": addr,
			"path": path,
		})
		if err := c.httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.ErrorCF("messenger", "Webhook server error", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}()

	c.setRunning(true)
	logger.InfoC("messenger", "Messenger channel started (Webhook Mode)")
	return nil
}

// Stop gracefully shuts down the HTTP server.
func (c *MessengerChannel) Stop(ctx context.Context) error {
	logger.InfoC("messenger", "Stopping Messenger channel")

	if c.cancel != nil {
		c.cancel()
	}

	if c.httpServer != nil {
		shutdownCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
		defer cancel()
		if err := c.httpServer.Shutdown(shutdownCtx); err != nil {
			logger.ErrorCF("messenger", "Webhook server shutdown error", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}

	c.setRunning(false)
	logger.InfoC("messenger", "Messenger channel stopped")
	return nil
}

// webhookHandler handles incoming Messenger webhook requests.
// GET requests handle webhook verification; POST requests handle incoming messages.
func (c *MessengerChannel) webhookHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		c.handleVerification(w, r)
	case http.MethodPost:
		c.handleIncoming(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleVerification responds to the Meta webhook verification challenge.
func (c *MessengerChannel) handleVerification(w http.ResponseWriter, r *http.Request) {
	mode := r.URL.Query().Get("hub.mode")
	token := r.URL.Query().Get("hub.verify_token")
	challenge := r.URL.Query().Get("hub.challenge")

	if mode == "subscribe" && token == c.config.VerifyToken {
		logger.InfoC("messenger", "Webhook verification successful")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(challenge))
		return
	}

	logger.WarnC("messenger", "Webhook verification failed")
	http.Error(w, "Forbidden", http.StatusForbidden)
}

// handleIncoming processes incoming webhook POST requests.
func (c *MessengerChannel) handleIncoming(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		logger.ErrorCF("messenger", "Failed to read request body", map[string]interface{}{
			"error": err.Error(),
		})
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Verify signature
	signature := r.Header.Get("X-Hub-Signature-256")
	if !c.verifySignature(body, signature) {
		logger.WarnC("messenger", "Invalid webhook signature")
		http.Error(w, "Forbidden", http.StatusForbidden)
		return
	}

	var payload messengerWebhookPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		logger.ErrorCF("messenger", "Failed to parse webhook payload", map[string]interface{}{
			"error": err.Error(),
		})
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	// Return 200 immediately, process events asynchronously
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("EVENT_RECEIVED"))

	for _, entry := range payload.Entry {
		for _, event := range entry.Messaging {
			go c.processEvent(event)
		}
	}
}

// verifySignature validates the X-Hub-Signature-256 using HMAC-SHA256.
func (c *MessengerChannel) verifySignature(body []byte, signature string) bool {
	if signature == "" {
		return false
	}

	// Signature format: "sha256=<hex>"
	prefix := "sha256="
	if !strings.HasPrefix(signature, prefix) {
		return false
	}
	sigHex := signature[len(prefix):]

	mac := hmac.New(sha256.New, []byte(c.config.AppSecret))
	mac.Write(body)
	expected := hex.EncodeToString(mac.Sum(nil))

	return hmac.Equal([]byte(expected), []byte(sigHex))
}

// Messenger webhook payload types
type messengerWebhookPayload struct {
	Object string                   `json:"object"`
	Entry  []messengerWebhookEntry  `json:"entry"`
}

type messengerWebhookEntry struct {
	ID        string                    `json:"id"`
	Time      int64                     `json:"time"`
	Messaging []messengerMessagingEvent `json:"messaging"`
}

type messengerMessagingEvent struct {
	Sender    messengerUser    `json:"sender"`
	Recipient messengerUser    `json:"recipient"`
	Timestamp int64            `json:"timestamp"`
	Message   *messengerMsg    `json:"message,omitempty"`
}

type messengerUser struct {
	ID string `json:"id"`
}

type messengerMsg struct {
	MID         string                    `json:"mid"`
	Text        string                    `json:"text"`
	Attachments []messengerAttachment     `json:"attachments,omitempty"`
}

type messengerAttachment struct {
	Type    string                    `json:"type"`
	Payload messengerAttachmentPayload `json:"payload"`
}

type messengerAttachmentPayload struct {
	URL string `json:"url"`
}

func (c *MessengerChannel) processEvent(event messengerMessagingEvent) {
	// Only process message events
	if event.Message == nil {
		logger.DebugCF("messenger", "Ignoring non-message event", map[string]interface{}{
			"sender": event.Sender.ID,
		})
		return
	}

	senderID := event.Sender.ID
	chatID := senderID // Messenger conversations are 1:1 with sender

	var content string

	if event.Message.Text != "" {
		content = event.Message.Text
	} else if len(event.Message.Attachments) > 0 {
		// Log attachments but skip for now
		for _, att := range event.Message.Attachments {
			logger.InfoCF("messenger", "Received attachment (skipping)", map[string]interface{}{
				"type": att.Type,
				"url":  att.Payload.URL,
			})
		}
		content = fmt.Sprintf("[%s attachment]", event.Message.Attachments[0].Type)
	}

	if strings.TrimSpace(content) == "" {
		return
	}

	metadata := map[string]string{
		"platform":   "messenger",
		"message_id": event.Message.MID,
	}

	logger.DebugCF("messenger", "Received message", map[string]interface{}{
		"sender_id": senderID,
		"chat_id":   chatID,
		"preview":   utils.Truncate(content, 50),
	})

	c.HandleMessage(senderID, chatID, content, nil, metadata)
}

// Send sends a message to a Messenger user via the Graph API.
func (c *MessengerChannel) Send(ctx context.Context, msg bus.OutboundMessage) error {
	if !c.IsRunning() {
		return fmt.Errorf("messenger channel not running")
	}

	payload := map[string]interface{}{
		"recipient": map[string]string{
			"id": msg.ChatID,
		},
		"message": map[string]string{
			"text": msg.Content,
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	url := messengerMessagesAPI + "?access_token=" + c.config.PageAccessToken
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Graph API request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("Messenger Graph API error (status %d): %s", resp.StatusCode, string(respBody))
	}

	logger.DebugCF("messenger", "Message sent", map[string]interface{}{
		"chat_id": msg.ChatID,
	})

	return nil
}
