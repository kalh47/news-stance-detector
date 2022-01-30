// import Math;
// HTML elements for injection
disagreehtml = ' <img src="http://www.clker.com/cliparts/P/Z/w/n/R/W/red-smiley-face-hi.png" title="MISLEADING HEADLINE:\n' +
    'Automatic Stance Detection says this headline disagrees with the article.\n' +
    '(Note this tool showed **% accuracy during testing)" width="40" height="40">';
unrelatedhtml = ' <img src="http://www.clker.com/cliparts/P/Z/w/n/R/W/red-smiley-face-hi.png" title="UNRELATED HEADLINE:\n' +
    'Automatic Stance Detection says this headline is unrelated to the article.\n' +
    '(Note this tool showed **% accuracy during testing)" width="40" height="40">';
discussedhtml = ' <img src="http://www.clker.com/cliparts/I/X/g/L/q/2/yellow-neutral-face-md.png" title="DISCUSSED HEADLINE:\n' +
    'Automatic Stance Detection says this headline is discussed in the article.\n' +
    '(Note this tool showed **% accuracy during testing)" width="40" height="40">';
agreehtml = ' <img src="http://www.clker.com/cliparts/0/b/c/1/120657385441132693Arnoud999_Right_or_wrong_2.svg.med.png" title="AGREEING HEADLINE:\n' +
    'Automatic Stance Detection says this headline agrees with the article.\n' +
    '(Note this tool showed **% accuracy during testing)" width="40" height="40">';

predictions = ["Disagree","Unrelated","Discuss","Agree"]
outcomes = [disagreehtml,unrelatedhtml,discussedhtml,agreehtml]
// alert('headline '+document.getElementsByTagName("h1")[0].innerText);
// $.ajax({
//     type: "POST",
//     url: "./testrandom.py",
//     data: { param: document}
// }).done(function( o ) {
//     // do something
//     alert(result)
//     document.getElementsByTagName("h1")[0].innerHTML = document.getElementsByTagName("h1")[0].innerHTML + disagreehtml;
// });

// Injection
let num = Math.floor(Math.random() * 4); //temporary random number stance to test each HTML injection
document.getElementsByTagName("h1")[0].innerHTML = document.getElementsByTagName("h1")[0].innerHTML + outcomes[num];

// let scrapeJSON = '~/../Mydetector/prep.py'
// $.get(scrapeJSON, function(data) {
//     // Get JSON data from Python script
//     if (data){
//         console.log("Data returned:", data)
//     }
//     jobDataJSON = JSON.parse(data)
// })


//(powershell) python -m http.server 8080
$.ajax({
    type: "POST",
    url: "http://localhost:8080/prep.py",
    data: {}
}).done(function( o ) {
    // do something
    var test_headline, test_body = scrape("https://www.bbc.co.uk/news/uk-england-london-55730459")
    console.log(test_headline)
});