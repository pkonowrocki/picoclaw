package channels

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sipeed/picoclaw/pkg/bus"
	"github.com/sipeed/picoclaw/pkg/config"
)

func newTestMessengerChannel(t *testing.T) *MessengerChannel {
	t.Helper()
	cfg := config.MessengerConfig{
		Enabled:         true,
		PageAccessToken: "test-page-token",
		VerifyToken:     "test-verify-token",
		AppSecret:       "test-app-secret",
		WebhookPort:     0,
		WebhookPath:     "/messenger/webhook",
	}
	msgBus := bus.NewMessageBus()
	ch, err := NewMessengerChannel(cfg, msgBus)
	if err != nil {
		t.Fatalf("NewMessengerChannel: %v", err)
	}
	return ch
}

func computeSignature(secret, body string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(body))
	return "sha256=" + hex.EncodeToString(mac.Sum(nil))
}

func TestMessengerVerifySignature(t *testing.T) {
	ch := newTestMessengerChannel(t)

	tests := []struct {
		name      string
		body      string
		signature string
		want      bool
	}{
		{
			name:      "valid signature",
			body:      `{"object":"page"}`,
			signature: computeSignature("test-app-secret", `{"object":"page"}`),
			want:      true,
		},
		{
			name:      "invalid signature",
			body:      `{"object":"page"}`,
			signature: "sha256=0000000000000000000000000000000000000000000000000000000000000000",
			want:      false,
		},
		{
			name:      "empty signature",
			body:      `{"object":"page"}`,
			signature: "",
			want:      false,
		},
		{
			name:      "missing sha256 prefix",
			body:      `{"object":"page"}`,
			signature: "deadbeef",
			want:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ch.verifySignature([]byte(tt.body), tt.signature)
			if got != tt.want {
				t.Errorf("verifySignature() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMessengerWebhookVerification(t *testing.T) {
	ch := newTestMessengerChannel(t)

	tests := []struct {
		name       string
		query      string
		wantStatus int
		wantBody   string
	}{
		{
			name:       "valid verification",
			query:      "hub.mode=subscribe&hub.verify_token=test-verify-token&hub.challenge=challenge123",
			wantStatus: http.StatusOK,
			wantBody:   "challenge123",
		},
		{
			name:       "wrong verify token",
			query:      "hub.mode=subscribe&hub.verify_token=wrong-token&hub.challenge=challenge123",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "wrong mode",
			query:      "hub.mode=unsubscribe&hub.verify_token=test-verify-token&hub.challenge=challenge123",
			wantStatus: http.StatusForbidden,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/messenger/webhook?"+tt.query, nil)
			w := httptest.NewRecorder()
			ch.webhookHandler(w, req)

			resp := w.Result()
			if resp.StatusCode != tt.wantStatus {
				t.Errorf("status = %d, want %d", resp.StatusCode, tt.wantStatus)
			}
			if tt.wantBody != "" {
				body, _ := io.ReadAll(resp.Body)
				if strings.TrimSpace(string(body)) != tt.wantBody {
					t.Errorf("body = %q, want %q", string(body), tt.wantBody)
				}
			}
		})
	}
}

func TestMessengerMessageParsing(t *testing.T) {
	ch := newTestMessengerChannel(t)

	payload := messengerWebhookPayload{
		Object: "page",
		Entry: []messengerWebhookEntry{
			{
				ID:   "page123",
				Time: 1234567890,
				Messaging: []messengerMessagingEvent{
					{
						Sender:    messengerUser{ID: "user456"},
						Recipient: messengerUser{ID: "page123"},
						Timestamp: 1234567890,
						Message: &messengerMsg{
							MID:  "mid.123",
							Text: "Hello, bot!",
						},
					},
				},
			},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	signature := computeSignature("test-app-secret", string(body))

	req := httptest.NewRequest(http.MethodPost, "/messenger/webhook", strings.NewReader(string(body)))
	req.Header.Set("X-Hub-Signature-256", signature)
	w := httptest.NewRecorder()

	ch.webhookHandler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	respBody, _ := io.ReadAll(resp.Body)
	if strings.TrimSpace(string(respBody)) != "EVENT_RECEIVED" {
		t.Errorf("body = %q, want %q", string(respBody), "EVENT_RECEIVED")
	}
}

func TestMessengerRejectsInvalidSignature(t *testing.T) {
	ch := newTestMessengerChannel(t)

	body := `{"object":"page","entry":[]}`
	req := httptest.NewRequest(http.MethodPost, "/messenger/webhook", strings.NewReader(body))
	req.Header.Set("X-Hub-Signature-256", "sha256=invalid")
	w := httptest.NewRecorder()

	ch.webhookHandler(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusForbidden {
		t.Errorf("status = %d, want %d", resp.StatusCode, http.StatusForbidden)
	}
}

func TestNewMessengerChannelRequiresCredentials(t *testing.T) {
	msgBus := bus.NewMessageBus()

	_, err := NewMessengerChannel(config.MessengerConfig{
		Enabled: true,
	}, msgBus)
	if err == nil {
		t.Error("expected error for missing credentials")
	}

	_, err = NewMessengerChannel(config.MessengerConfig{
		Enabled:         true,
		PageAccessToken: "token",
		AppSecret:       "secret",
	}, msgBus)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
