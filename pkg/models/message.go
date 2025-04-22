package models

import "time"

// Role represents the role of a message sender
type Role string

const (
	// RoleUser represents a message from the user
	RoleUser Role = "user"
	// RoleAssistant represents a message from the assistant
	RoleAssistant Role = "assistant"
	// RoleSystem represents a system message
	RoleSystem Role = "system"
)

// Message represents a chat message
type Message struct {
	Role      Role      `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// Chat represents a complete conversation with multiple messages
type Chat struct {
	ID       string    `json:"id"`
	Messages []Message `json:"messages"`
	Created  time.Time `json:"created"`
	Updated  time.Time `json:"updated"`
}
