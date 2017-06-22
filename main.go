package main

import (
  "fmt"
  "log"
  "io/ioutil"
  "strings"
  "strconv"
  tf "github.com/tensorflow/tensorflow/tensorflow/go"
)


func main(){
  sm, err := tf.LoadSavedModel("./tensorflow-export",
      []string{"serve"},nil)
  if err != nil {
    log.Fatal(err)
  }
  data, err := data_from_file("single_img")
  if err != nil {
    log.Fatal(err)
  }
  img_data, err := process_img(data)
  if err != nil {
    log.Fatal(err)
  }
  tensor := load_data(img_data)
  if _,err := exec_model(sm,tensor); err != nil {
    log.Fatal(err)
  }
}

func exec_model(sm *tf.SavedModel,input *tf.Tensor) (string, error){
  sess := sm.Session
  grph := sm.Graph
  output, err := sess.Run(map[tf.Output]*tf.Tensor{
    grph.Operation("input").Output(0): input,
  },
  []tf.Output{
    grph.Operation("output").Output(0),
  },
  nil)
  if err != nil {
    return "", err
  }
  prob := output[0].Value().([][]float32)[0]
  fmt.Printf("%T %V",prob,prob)
  return "",nil
}

func load_data(data [][]float32) *tf.Tensor{
  tensor, err := tf.NewTensor(data)
  if err != nil {
    log.Fatal(err)
  }
  return tensor
}

func process_img(data [][]int) ([][]float32,error) {
  out := [][]float32 {}
  for _,v := range data {
    obj := []float32 {}
    for _,x := range v {
      obj = append(obj,(float32(x) / 255))
    }
    out = append(out,obj)
  }
  return out,nil
}

func data_from_file(fname string) ([][]int, error){
  bytes, err := ioutil.ReadFile(fname)
  if err != nil {
    return nil, err
  }
  s := string(bytes[:])
  out := [][]int{}
  strings.Split(s,"\n")
  for _,l := range strings.Split(s,"\n") {
    chunk := []int {}
    for _,c := range strings.Split(l," ") {
      var val,_ = strconv.Atoi(c)
      chunk = append(chunk,val)
    }
    out = append(out,chunk)
  }
  return out,nil
}
