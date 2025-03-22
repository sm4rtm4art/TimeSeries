import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface D3ChartProps {
  data: any;
}

const D3Chart: React.FC<D3ChartProps> = ({ data }) => {
  const chartRef = useRef(null);

  useEffect(() => {
    if (data) {
      drawChart();
    }
  }, [data]);

  const drawChart = () => {
    // Clear existing chart
    d3.select(chartRef.current).selectAll("*").remove();

    // Set up dimensions
    const width = 600;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 30, left: 40 };

    // Create SVG
    const svg = d3.select(chartRef.current)
      .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Parse data
    const parsedData = Object.entries(data).flatMap(([model, forecast]) =>
      forecast.map((d) => ({ model, date: new Date(d.date), value: d.value }))
    );

    // Set up scales
    const x = d3.scaleTime()
      .domain(d3.extent(parsedData, d => d.date))
      .range([0, width]);

    const y = d3.scaleLinear()
      .domain([0, d3.max(parsedData, d => d.value)])
      .range([height, 0]);

    // Add axes
    svg.append("g")
      .attr("transform", `translate(0,${height})`)
      .call(d3.axisBottom(x));

    svg.append("g")
      .call(d3.axisLeft(y));

    // Add lines
    const line = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.value));

    const color = d3.scaleOrdinal(d3.schemeCategory10);

    Object.keys(data).forEach((model) => {
      svg.append("path")
        .datum(parsedData.filter(d => d.model === model))
        .attr("fill", "none")
        .attr("stroke", color(model))
        .attr("stroke-width", 1.5)
        .attr("d", line);
    });

    // Add legend
    const legend = svg.selectAll(".legend")
      .data(Object.keys(data))
      .enter().append("g")
      .attr("class", "legend")
      .attr("transform", (d, i) => `translate(0,${i * 20})`);

    legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", color);

    legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(d => d);
  };

  return <div ref={chartRef}></div>;
};

export default D3Chart;
