// Student Performance Dashboard

document.addEventListener("DOMContentLoaded", function () {

    console.log("Student Performance Dashboard Loaded Successfully!");

    // Highlight topper row
    const tables = document.querySelectorAll("table");

    if (tables.length > 0) {
        const topperTable = tables[0];

        if (topperTable.rows.length > 1) {
            topperTable.rows[1].style.backgroundColor = "#d4edda";
            topperTable.rows[1].style.fontWeight = "bold";
        }
    }

});