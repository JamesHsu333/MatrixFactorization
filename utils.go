package main

import (
	"bufio"
	"encoding/csv"
	"gonum.org/v1/gonum/mat"
	"io"
	"os"
	"strconv"
)

type Dataset struct {
	Data  *mat.Dense
	Label *mat.Dense
}

func NewDataset(features []float64, targets []float64) *Dataset {
	data := mat.NewDense(len(targets), len(features)/len(targets), features)
	label := mat.NewDense(1, len(targets), targets)
	return &Dataset{
		Data:  data,
		Label: label,
	}
}

func readData(filename string, length int) ([]float64, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	reader := csv.NewReader(bufio.NewReader(file))

	var targets []float64
	var features []float64

	for {
		data, err := reader.Read()
		if err == io.EOF {
			break
		}

		target, err := strconv.ParseFloat(data[0], 64)
		if err != nil {
			return nil, nil, err
		}
		targets = append(targets, target)

		for i := 1; i < len(data); i += 1 {
			res, err := strconv.ParseFloat(data[i], 64)
			if err != nil {
				return nil, nil, err
			}
			features = append(features, res)
		}
	}
	return features, targets, nil
}
