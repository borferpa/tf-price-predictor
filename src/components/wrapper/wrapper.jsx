import React, { useEffect, useRef, useState } from "react";
import {
    Chart,
    LinearScale,
    PointElement,
    LineElement,
    Tooltip,
    Legend,
  } from 'chart.js';
  import { Scatter } from 'react-chartjs-2';
  import dataTransform from "../../utils/dataTransform";
import * as tf from "@tensorflow/tfjs";
import styles from './wrapper.scss';


const Wrapper = () => {
    Chart.register(LinearScale, PointElement, LineElement, Tooltip, Legend);


    const salesByWeek = [
      {
        sale: 100
      },
      {
        sale: 110
      },
      {
        sale: 80
      },
      {
        sale: 90
      },
      {
        sale: 150
      },
      {
        sale: 130
      },
    ];

    const weeks = dataTransform().getWeeks(salesByWeek);
    const sales = dataTransform().getSales(salesByWeek);
    const dataModel = dataTransform().convertToScatterModel(salesByWeek);

    const [data, setData] = useState({
      datasets: [
          {
              label: 'Ventas por semana',
              data: dataModel,
              backgroundColor: 'pink',
          }
      ]
    });

    const options = {
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    };

    const predictSales = async () => {
      const model = tf.sequential();
      model.add(tf.layers.dense({
        units: 1,
        inputShape: [1],
      }));
      model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd',
      });

      const weeksTensor = tf.tensor2d(weeks, [weeks.length, 1]);
      const salesTensor = tf.tensor2d(sales, [sales.length, 1]);
      const epochs = 2000;
      const weekToPredict = 7;

      for(let i = 0; i < epochs; i++) {
        await model.fit(weeksTensor, salesTensor, {epochs: 1});
        const predictionSale = model.predict(tf.tensor2d([weekToPredict], [1, 1])).dataSync()[0];

        dataModel.push({x: weekToPredict, y: predictionSale});
        setData({
          datasets: [
              {
                  label: 'Ventas por semana',
                  data: dataModel,
              }
          ]
      })
        console.log(dataModel);
      }
    };

      useEffect(() => {
        predictSales();
      }, [])

    return (
            <Scatter options={options} data={data} />
    );
};

export default Wrapper;