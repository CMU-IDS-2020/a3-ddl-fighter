# Project name

![A screenshot of your application. Could be a GIF.](screenshot.png)

TODO: Short abstract describing the main goals and how you achieved them.

## Project Goals

TODO: **A clear description of the goals of your project.** Describe the question that you are enabling a user to answer. The question should be compelling and the solution should be focused on helping users achieve their goals. 

## Design

TODO: **A rationale for your design decisions.** How did you choose your particular visual encodings and interaction techniques? What alternatives did you consider and how did you arrive at your ultimate choices?



We explore categories and business reaction in Section 3, especially in terms of **Temporary Closed Until**, **Covid Banner** and **highlights**. We use color to encode categories, because 1) color is recommended for encoding norminal features and 2) we mainly want to use stacked bar charts to show a decomposition of certain reaction into categories. For interactivity, we allow readers to select target COVID features as we have done before. Meanwhile, in both section 3.1&3, we allow readers to select time window to explore in more detail. When reviewing our design, we assume readers may be interested in comparison within a category, and therefore additionally for 3.3, we support highlights bars of a selected category.

For section 4.1, we explore how quality affects interaction. Here we choose the scatter plot given that we have two quantitative variables and one norminal. We start by overviewing rating stars and review counts. We initially made it a static graph, but later we assume readers might want to see how these two factors interact, and thus support intersected interval selection. In 4.2, we choose the scatter plotallow readers to "design" their target scatter plot, with deciding a target, a sample size and a filter. Due to the rating stars are discrete, the plot could be crowded. Therefore, we additional support highlight on selection.


## Development

TODO: **An overview of your development process.** Describe how the work was split among the team members. Include a commentary on the development process, including answers to the following questions: Roughly how much time did you spend developing your application (in people-hours)? What aspects took the most time?

took the most time!!!!!! 

Group data into certain formation through Pandas and Numpy because altair does not quite support huge dataset.
Also, adding interactive widgets for altair is tricky, with some operations leading to unexpected effects or errors.
