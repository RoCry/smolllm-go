package smolllm

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChoosePairPrefersLeastUsedSingleURL(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		mu: sync.Mutex{},
		usage: map[pairKey]int{
			{Key: "k1", URL: "u1"}: 2,
			{Key: "k2", URL: "u1"}: 2,
		},
		rnd: rand.New(rand.NewSource(42)),
	}

	key, url, err := b.choosePair("k1,k2,k3", "u1")
	require.NoError(t, err)
	assert.Equal(t, "u1", url)
	assert.Equal(t, "k3", key)
	assert.Equal(t, 1, b.usage[pairKey{Key: "k3", URL: "u1"}])
}

func TestChoosePairWithPairedKeysAndURLs(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		mu: sync.Mutex{},
		usage: map[pairKey]int{
			{Key: "k1", URL: "u1"}: 5,
			{Key: "k2", URL: "u2"}: 0,
		},
		rnd: rand.New(rand.NewSource(7)),
	}

	key, url, err := b.choosePair("k1,k2", "u1,u2")
	require.NoError(t, err)
	assert.Equal(t, "k2", key)
	assert.Equal(t, "u2", url)
	assert.Equal(t, 1, b.usage[pairKey{Key: "k2", URL: "u2"}])
}

func TestChoosePairValidation(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		keys    string
		urls    string
		usage   map[pairKey]int
		wantKey string
		wantURL string
		wantErr string
	}{
		{
			name:    "single key and url",
			keys:    "k1",
			urls:    "u1",
			usage:   nil,
			wantKey: "k1",
			wantURL: "u1",
			wantErr: "",
		},
		{
			name: "multiple keys single url picks least used",
			keys: "k1,k2",
			urls: "u1",
			usage: map[pairKey]int{
				{Key: "k1", URL: "u1"}: 3,
			},
			wantKey: "k2",
			wantURL: "u1",
			wantErr: "",
		},
		{
			name:    "mismatched counts",
			keys:    "a,b",
			urls:    "u1,u2,u3",
			usage:   nil,
			wantKey: "",
			wantURL: "",
			wantErr: "counts must match",
		},
		{
			name:    "empty entry rejected",
			keys:    "a,",
			urls:    "u1",
			usage:   nil,
			wantKey: "",
			wantURL: "",
			wantErr: "empty entry",
		},
		{
			name:    "blank urls string",
			keys:    "k1",
			urls:    "",
			usage:   nil,
			wantKey: "",
			wantURL: "",
			wantErr: "must not be empty",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			b := &simpleBalancer{
				mu:    sync.Mutex{},
				usage: make(map[pairKey]int, len(tc.usage)),
				rnd:   rand.New(rand.NewSource(1)),
			}
			for key, usage := range tc.usage {
				b.usage[key] = usage
			}

			key, url, err := b.choosePair(tc.keys, tc.urls)
			if tc.wantErr != "" {
				require.Error(t, err)
				require.ErrorContains(t, err, tc.wantErr)
				assert.Empty(t, key)
				assert.Empty(t, url)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tc.wantKey, key)
			assert.Equal(t, tc.wantURL, url)
		})
	}
}
