package smolllm

import (
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"time"
)

type pairKey struct {
	Key string
	URL string
}

type simpleBalancer struct {
	mu    sync.Mutex
	usage map[pairKey]int
	rnd   *rand.Rand
}

var balancer = &simpleBalancer{
	mu:    sync.Mutex{},
	usage: make(map[pairKey]int),
	rnd:   rand.New(rand.NewSource(time.Now().UTC().UnixNano())),
}

func (b *simpleBalancer) choosePair(keys string, urls string) (string, string, error) {
	pairs, err := buildPairs(keys, urls)
	if err != nil {
		return "", "", err
	}

	b.mu.Lock()
	defer b.mu.Unlock()

	minUsage := -1
	var least []pairKey
	for _, pair := range pairs {
		usage := b.usage[pair]
		if minUsage == -1 || usage < minUsage {
			minUsage = usage
			least = least[:0]
			least = append(least, pair)
			continue
		}
		if usage == minUsage {
			least = append(least, pair)
		}
	}

	if len(least) == 0 {
		return "", "", fmt.Errorf("no eligible key/url pair")
	}

	if b.rnd == nil {
		return "", "", fmt.Errorf("random source not configured")
	}

	chosen := least[b.rnd.Intn(len(least))]
	b.usage[chosen]++

	return chosen.Key, chosen.URL, nil
}

func buildPairs(keys string, urls string) ([]pairKey, error) {
	keyList, err := parseList(keys)
	if err != nil {
		return nil, err
	}
	urlList, err := parseList(urls)
	if err != nil {
		return nil, err
	}

	if len(keyList) == 0 || len(urlList) == 0 {
		return nil, fmt.Errorf("API key and base URL must be non-empty")
	}

	switch {
	case len(urlList) == 1:
		pairs := make([]pairKey, 0, len(keyList))
		for _, key := range keyList {
			pairs = append(pairs, pairKey{Key: key, URL: urlList[0]})
		}
		return pairs, nil
	case len(keyList) == 1:
		pairs := make([]pairKey, 0, len(urlList))
		for _, url := range urlList {
			pairs = append(pairs, pairKey{Key: keyList[0], URL: url})
		}
		return pairs, nil
	default:
		if len(keyList) != len(urlList) {
			return nil, fmt.Errorf("when using multiple keys and URLs, counts must match")
		}
		pairs := make([]pairKey, 0, len(keyList))
		for i := range keyList {
			pairs = append(pairs, pairKey{Key: keyList[i], URL: urlList[i]})
		}
		return pairs, nil
	}
}

func parseList(items string) ([]string, error) {
	if strings.TrimSpace(items) == "" {
		return nil, fmt.Errorf("value must not be empty")
	}
	raw := strings.Split(items, ",")
	result := make([]string, 0, len(raw))
	for _, item := range raw {
		value := strings.TrimSpace(item)
		if value == "" {
			return nil, fmt.Errorf("list contains empty entry")
		}
		result = append(result, value)
	}
	return result, nil
}

func validateKeyURLPairs(keys string, urls string) error {
	_, err := buildPairs(keys, urls)
	return err
}
