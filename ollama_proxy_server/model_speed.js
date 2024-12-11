$(function() {
    $('#date-range').daterangepicker({
        opens: 'left',
        startDate: "2024/01/01",
        endDate: "2024/12/31",
        locale: {
            format: 'YYYY-MM-DD',
            separator: " - ",
            cancelLabel: "Cancel",
        },
        ranges: {
            'Today': [moment(), moment()],
            'Yesterday': [moment().subtract(1, 'days'), moment().subtract(1, 'days')],
            'Last 7 Days': [moment().subtract(6, 'days'), moment()],
            'Last 30 Days': [moment().subtract(29, 'days'), moment()],
            'This Month': [moment().startOf('month'), moment().endOf('month')],
            'Last Month': [moment().subtract(1, 'month').startOf('month'), moment().subtract(1, 'month').endOf('month')],
            'All Time': [moment("2024-01-01"), moment()]
        },
        timePicker: true,
        alwaysShowCalendars: true,
        timePicker24Hour: true,
    });

    $("#model-server-select").select2();

    const fetchData = async () => {
        try {
            const response = await fetch('/local/model_stats'); // Your actual data endpoint
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching data:', error);
            return null;
        }
    };

    const populateModelServerPicker = (data) => {
        const combinations = new Set();

        Object.keys(data.time_stamp).forEach(key => {
            const model = data.model[key];
            const server = data.server[key];
            combinations.add(`${model} - ${server}`);
        });

        const select = $("#model-server-select");
        select.empty();

        combinations.forEach(combination => {
            select.append(new Option(combination, combination));
        });

        // Select all options by default
        select.val(Array.from(combinations)).trigger('change');
    };

    const processData = (data, startDate, endDate, selectedCombinations) => {
        return Object.keys(data.time_stamp).filter(key => {
            const date = new Date(data.time_stamp[key]);
            const modelServer = `${data.model[key]} - ${data.server[key]}`;
            return (!startDate || date >= new Date(startDate)) &&
                   (!endDate || date <= new Date(endDate + "T23:59:59")) &&
                   selectedCombinations.includes(modelServer);
        }).map(key => ({
            time_stamp: data.time_stamp[key],
            total_speed: data.total_speed[key],
            prompt_eval_speed: data.prompt_eval_speed[key],
            eval_speed: data.eval_speed[key]
        }));
    };

    const renderChart = (chartData) => {
        // Filter out data points where any of the speed values exceed 2000
        const filteredData = chartData.filter(item =>
            item.total_speed <= 2000 &&
            item.eval_speed <= 2000 &&
            item.prompt_eval_speed <= 2000
        );

                    // Calculate averages
        const totalSpeedAvg = filteredData.reduce((sum, item) => sum + item.total_speed, 0) / filteredData.length;
        const evalSpeedAvg = filteredData.reduce((sum, item) => sum + item.eval_speed, 0) / filteredData.length;
        const promptEvalSpeedAvg = filteredData.reduce((sum, item) => sum + item.prompt_eval_speed, 0) / filteredData.length;

        // Update the textarea with average speeds
        document.getElementById("average-speeds").value = `
            Average Total Speed: ${totalSpeedAvg.toFixed(2)} tokens/s\n
            Average Eval Speed: ${evalSpeedAvg.toFixed(2)} tokens/s\n
            Average Prompt Eval Speed: ${promptEvalSpeedAvg.toFixed(2)}  tokens/s
        `;

        Highcharts.chart('container', {
            title: {
                text: 'Speed Data Scatter Plot'
            },
            xAxis: {
                type: 'datetime',
                title: {
                    text: 'Time'
                }
            },
            yAxis: [{
                // Primary Y-axis for total_speed and eval_speed
                title: {
                    text: 'Speed (tokens/s)'
                },
                opposite: false  // Placed on the left (default)
            }, {
                // Secondary Y-axis for prompt_eval_speed
                title: {
                    text: 'Prompt Eval Speed (tokens/s)'
                },
                opposite: true  // Placed on the right
            }],
            tooltip: {
                shared: true,
                crosshairs: true,
                // Format the x-axis (time) as a human-readable date
                xDateFormat: '%Y-%m-%d %H:%M:%S',  // You can adjust the format here
                pointFormat: '<span>{series.name}</span>: <b>{point.y}</b><br/>'
            },
            series: [
                {
                    name: 'Total Speed',
                    data: filteredData.map(item => [new Date(item.time_stamp).getTime(), item.total_speed]),
                    type: 'scatter',
                    color: '#0071A7',
                    yAxis: 0 // This series will use the first Y-axis
                },
                {
                    name: 'Eval Speed',
                    data: filteredData.map(item => [new Date(item.time_stamp).getTime(), item.eval_speed]),
                    type: 'scatter',
                    color: '#28a745',
                    yAxis: 0 // This series will use the first Y-axis
                },
                {
                    name: 'Prompt Eval Speed',
                    data: filteredData.map(item => [new Date(item.time_stamp).getTime(), item.prompt_eval_speed]),
                    type: 'scatter',
                    color: '#FFA500',
                    yAxis: 1 // This series will use the second Y-axis
                }
            ]
        });
    };

    const updateChart = async () => {
        const dateRange = $("#date-range").val().split(" - ");
        const startDate = dateRange[0];
        const endDate = dateRange[1];
        const selectedCombinations = $("#model-server-select").val();
        const data = await fetchData();

        if (data) {
            const chartData = processData(data, startDate, endDate, selectedCombinations);
            renderChart(chartData);
        }
    };

    $('#date-range').on('apply.daterangepicker', updateChart);
    $("#model-server-select").on("change", updateChart);

    const initialize = async () => {
        const data = await fetchData();
        if (data) {
            populateModelServerPicker(data);  // Populate and select all combinations
            updateChart();  // Initial chart load
        }
    };

    initialize();
});
