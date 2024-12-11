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

    // Initialize Select2 with multiple selection
    $("#model-server-select").select2({
        placeholder: "Select Model and Server",
        width: '100%',
        closeOnSelect: false
    });

    const fetchData = async () => {
        try {
            const response = await fetch('/local/model_stats');
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
        select.val(Array.from(combinations)); // Select all combinations
        select.trigger('change'); // Trigger change to update the chart
    };

    const processData = (data, startDate, endDate, selectedCombinations) => {
        const filteredData = [];

        // Filter and aggregate data across all selected combinations
        Object.keys(data.time_stamp).forEach(key => {
            const date = new Date(data.time_stamp[key]);
            const modelServer = `${data.model[key]} - ${data.server[key]}`;

            if (
                (!startDate || date >= new Date(startDate)) &&
                (!endDate || date <= new Date(endDate + "T23:59:59")) &&
                selectedCombinations.includes(modelServer)
            ) {
                filteredData.push({
                    time_stamp: data.time_stamp[key],
                    input_tokens: data.input_tokens[key],
                    output_tokens: data.output_tokens[key]
                });
            }
        });

        // Sort data by time_stamp to ensure chronological order
        filteredData.sort((a, b) => new Date(a.time_stamp) - new Date(b.time_stamp));

        let cumulativeInput = 0;
        let cumulativeOutput = 0;

        return filteredData.map(item => {
            // Add the current tokens to the cumulative sum
            cumulativeInput += item.input_tokens;
            cumulativeOutput += item.output_tokens;

            return {
                time_stamp: item.time_stamp,
                cumulativeInput,
                cumulativeOutput
            };
        });
    };

    const renderChart = (chartData) => {
        Highcharts.chart('container', {
            title: {
                text: 'Cumulative Input and Output Tokens'
            },
            xAxis: {
                type: 'datetime',
                title: {
                    text: 'Time'
                }
            },
            yAxis: {
                title: {
                    text: 'Cumulative Tokens'
                }
            },
            tooltip: {
                shared: true,
                crosshairs: true
            },
            series: [
                {
                    name: 'Cumulative Input Tokens',
                    data: chartData.map(item => [new Date(item.time_stamp).getTime(), item.cumulativeInput]),
                    type: 'line',
                    color: '#0071A7'
                },
                {
                    name: 'Cumulative Output Tokens',
                    data: chartData.map(item => [new Date(item.time_stamp).getTime(), item.cumulativeOutput]),
                    type: 'line',
                    color: '#FFA500'
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

    // Trigger chart update when the date range changes
    $('#date-range').on('apply.daterangepicker', updateChart);

    // Trigger chart update when the model-server selection changes
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
