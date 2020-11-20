package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"os"
	"strconv"
)

type Training struct {
	Data  *mat.Dense
	Label *mat.Dense
}

type Testing struct {
	Data  *mat.Dense
	Label *mat.Dense
}

type Dataset interface {
	readData(string) error
}

func (t *Training) readData(filename string) error {
	if file, err := os.Open(filename); err != nil {
		return err
	}
	reader := csv.NewReader(bufio.NewReader(file))

	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}

		for i := 1; i < len(data); i += 1 {
			var features []float64
			if res, err := strconv.Parsefloat(data[i], 64); err != nil {
				return err
			}
			features = append(features, res)
		}
	}
}
