package smolllm

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestChoosePairPrefersLeastUsedSingleURL(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		usage: map[pairKey]int{
			{Key: "k1", URL: "u1"}: 2,
			{Key: "k2", URL: "u1"}: 2,
		},
		rnd: rand.New(rand.NewSource(42)),
	}

	key, url, err := b.choosePair("k1,k2,k3", "u1")
	require.NoError(t, err)
	require.Equal(t, "u1", url)
	require.Equal(t, "k3", key)
	require.Equal(t, 1, b.usage[pairKey{Key: "k3", URL: "u1"}])
}

func TestChoosePairWithPairedKeysAndURLs(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		usage: map[pairKey]int{
			{Key: "k1", URL: "u1"}: 5,
			{Key: "k2", URL: "u2"}: 0,
		},
		rnd: rand.New(rand.NewSource(7)),
	}

	key, url, err := b.choosePair("k1,k2", "u1,u2")
	require.NoError(t, err)
	require.Equal(t, "k2", key)
	require.Equal(t, "u2", url)
	require.Equal(t, 1, b.usage[pairKey{Key: "k2", URL: "u2"}])
}

func TestChoosePairErrorsOnMismatch(t *testing.T) {
	t.Parallel()
	b := &simpleBalancer{
		usage: make(map[pairKey]int),
		rnd:   rand.New(rand.NewSource(1)),
	}

	_, _, err := b.choosePair("k1,k2", "u1,u2,u3")
	require.Error(t, err)

	_, _, err = b.choosePair("k1,", "u1")
	require.Error(t, err)

	_, _, err = b.choosePair("k1", "")
	require.Error(t, err)
}
