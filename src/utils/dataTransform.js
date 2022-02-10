const dataTransform = () => {
    const convertToScatterModel = (salesByWeek) => {
        const scatterModel = [];
        salesByWeek.map((sale, week) => {
            scatterModel.push({
                x: week,
                y: sale.sale
            })
        });

        return scatterModel;
    };

    const getWeeks = (salesByWeek) => {
        const weeks = [];
        salesByWeek.map((sale, week) => {
            weeks.push(week);
        });
        return weeks;
    };

    const getSales = (salesByWeek) => {
        const sales = [];
        salesByWeek.map((sale, week) => {
            sales.push(sale.sale);
        });
        return sales;
    }

    return {
        convertToScatterModel,
        getWeeks,
        getSales,
    };
};

export default dataTransform;