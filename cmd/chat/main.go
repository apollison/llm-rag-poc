package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/andrew/llm-rag-poc/pkg/llm"
	"github.com/andrew/llm-rag-poc/pkg/models"
	"github.com/fatih/color"
)

var (
	modelName    = flag.String("model", "llama3", "Model name to use with Ollama")
	ollamaURL    = flag.String("ollama-url", "http://localhost:11434/api", "Ollama API URL")
	temperature  = flag.Float64("temp", 0.7, "Temperature for sampling")
	maxTokens    = flag.Int("max-tokens", 2048, "Maximum number of tokens to generate")
	systemPrompt = flag.String("system", "You are a helpful, honest, and concise assistant.", "System prompt")
)

func main() {
	flag.Parse()

	// Initialize context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle interrupts
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	go func() {
		<-c
		fmt.Println("\nShutting down...")
		cancel()
		os.Exit(0)
	}()

	// Initialize the Ollama client
	fmt.Printf("Initializing Ollama client with model: %s\n", *modelName)
	client := llm.NewOllamaClient(*modelName, *ollamaURL)

	// Print welcome message
	boldGreen := color.New(color.FgGreen, color.Bold).SprintFunc()
	boldCyan := color.New(color.FgCyan, color.Bold).SprintFunc()
	fmt.Println(boldGreen("🦙 Ollama Chat Interface"))
	fmt.Printf("Using model: %s\n", boldCyan(*modelName))
	fmt.Printf("Temperature: %.2f, Max Tokens: %d\n", *temperature, *maxTokens)
	fmt.Println("Type your message and press Enter. Type 'exit' or press Ctrl+C to quit.")
	fmt.Println()

	// Start chat session
	conversation := []models.Message{
		{
			Role:      models.RoleSystem,
			Content:   *systemPrompt,
			Timestamp: time.Now(),
		},
	}

	scanner := bufio.NewScanner(os.Stdin)
	modelConfig := llm.ModelConfig{
		Temperature: float32(*temperature),
		MaxTokens:   *maxTokens,
	}

	for {
		// Get user input
		fmt.Print(boldGreen("You: "))
		if !scanner.Scan() {
			break
		}
		userInput := scanner.Text()

		// Check for exit command
		if strings.ToLower(strings.TrimSpace(userInput)) == "exit" {
			break
		}

		// Add user message to conversation
		userMessage := models.Message{
			Role:      models.RoleUser,
			Content:   userInput,
			Timestamp: time.Now(),
		}
		conversation = append(conversation, userMessage)

		// Generate response
		fmt.Print(boldCyan("Assistant: "))
		response, err := client.Chat(ctx, conversation, modelConfig)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			fmt.Println("\nMake sure Ollama is running with: ollama serve")
			continue
		}

		// Output response
		fmt.Println(response.Content)
		fmt.Println()

		// Add assistant response to conversation history
		conversation = append(conversation, response)
	}
}
